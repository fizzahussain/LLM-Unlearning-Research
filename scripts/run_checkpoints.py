"""
Per-checkpoint pipeline for the LLM-unlearning project (notebook 04 logic).
Runs C0, C1-low, C1-high, C2, C3, C4 for each checkpoint, saving one
question-level CSV per condition and appending to logs/run_log.csv.
Restartable: a condition whose CSV already exists is skipped.

Usage (Kaggle cell, repo is the current directory):
    !python run_checkpoints.py --rows 13           # pilot: one SimNPO checkpoint
    !python run_checkpoints.py --rows 13-24        # your 12 checkpoints
    !python run_checkpoints.py --controls          # original + retain90 control
Add --operator member_2 (default) and --hf-token-env HF_TOKEN.
"""
import argparse, os, sys, time, json, hashlib, datetime, gc, random, subprocess
import numpy as np, pandas as pd, torch, yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer

p = argparse.ArgumentParser()
p.add_argument("--rows", default=None, help="manifest checkpoint_number, e.g. 13 or 13-24")
p.add_argument("--controls", action="store_true")
p.add_argument("--operator", default="member_2")
p.add_argument("--conditions", default="C0,C1-low,C1-high,C2,C3,C4")
p.add_argument("--limit", type=int, default=None, help="debug: only first N questions")
a = p.parse_args()

cfg = yaml.safe_load(open("configs/experiment_config.yaml"))
SEED = cfg["project"]["seed"]
assert SEED == 0, "config seed must be 0 (submitted document); fix configs/experiment_config.yaml"
c2 = cfg["conditions"].get("C2") or cfg["conditions"]["quantization"]   # v2: repo config names this block "quantization"
GS = c2.get("group_size")
assert GS, "add group_size (and symmetric: true/false) under conditions.C2 from paper [11] section 4.2"
SYM = bool(c2.get("symmetric", False))
L = cfg["lora"]; MAXLEN = L["max_length"]
GEN = cfg["generation"]
CFG_HASH = hashlib.md5(open("configs/experiment_config.yaml","rb").read()).hexdigest()[:8]
NB_VERSION = "run_checkpoints.py-v2"
dev = "cuda"
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
os.makedirs("results/per_question", exist_ok=True); os.makedirs("logs", exist_ok=True)

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ---------- data ----------
B = pd.read_csv("data/processed/part_b_evaluation.csv")
A = pd.read_csv("data/processed/part_a_relearning.csv")
R = pd.read_csv("data/processed/retain_evaluation.csv")
P = pd.read_csv("data/paraphrases/paraphrases.csv")
if a.limit: B, A, R, P = B.head(a.limit), A.head(a.limit), R.head(a.limit), P.head(a.limit)
Pm = P.merge(B[["question_id", "author_index"]], on="question_id")  # author index for C3

def build_mixed():
    path = "data/processed/mixed_prompts.csv"
    if os.path.exists(path): return pd.read_csv(path)
    rng = random.Random(SEED); rows = []
    r = R.sample(frac=1, random_state=SEED).reset_index(drop=True)
    for i, f in B.reset_index(drop=True).iterrows():
        rq = r.iloc[i % len(r)]
        for order in ["forget_first", "retain_first"]:
            q1, q2 = (f.question, rq.question) if order == "forget_first" else (rq.question, f.question)
            rows.append(dict(question_id=f.question_id, author_index=f.author_index, prompt_order=order,
                prompt=f"Answer both questions.\n1. {q1}\n2. {q2}",
                forget_answer=f.answer, retain_question_id=rq.question_id, retain_answer=rq.answer))
    d = pd.DataFrame(rows); d.to_csv(path, index=False); return d
M = build_mixed()

# v2: Truth Ratio needs TOFU's paraphrased and perturbed (wrong) answers for each forget question.
try:
    from datasets import load_dataset
    PT = load_dataset("locuslab/TOFU", "forget10_perturbed")["train"].to_pandas().drop_duplicates("question").set_index("question")
    print("forget10_perturbed columns:", list(PT.columns))
    print("Truth Ratio data loaded:", len(PT), "rows; part-B questions found:", int(B.question.isin(PT.index).sum()), "of", len(B))
except Exception as e:
    PT = None; print("WARNING: Truth Ratio data not loaded, truth_ratio will be skipped:", repr(e))

# ---------- model helpers ----------
def load(model_id, tok_id="open-unlearning/tofu_Llama-3.2-1B-Instruct_full"):
    tok = AutoTokenizer.from_pretrained(tok_id, token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16,
            token=os.environ.get("HF_TOKEN")).to(dev).eval()
    return tok, m

# v2: same prompt format as OpenUnlearning (configs/model/Llama-3.2-1B-Instruct.yaml):
# system prompt "You are a helpful assistant." and a FIXED date string, so runs on different days are identical.
SYSTEM_PROMPT = "You are a helpful assistant."
DATE_STRING = "10 Apr 2025"
def prompt_ids(tok, q):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, date_string=DATE_STRING)
    return list(ids["input_ids"]) if hasattr(ids, "keys") else list(ids)

def encode_qa(tok, q, ans):
    pi = prompt_ids(tok, q); ai = tok(ans + tok.eos_token, add_special_tokens=False).input_ids
    ids = (pi + ai)[:MAXLEN]; return ids, len(pi)

@torch.no_grad()
def tf_metrics(tok, m, qs, ans):
    """Teacher-forced metrics, one question at a time (memory-safe on T4)."""
    out = []
    for q, an in zip(qs, ans):
        ids, n = encode_qa(tok, q, an)
        x = torch.tensor([ids], device=dev)
        lg = m(x).logits[0, n - 1:-1].float()
        tgt = x[0, n:]
        lp = torch.log_softmax(lg, -1)
        tlp = lp.gather(1, tgt[:, None])[:, 0]
        ok = (lg.argmax(-1) == tgt).cpu().numpy()
        Lk = len(ok)
        # extraction strength: smallest k s.t. greedy decoding from first k answer tokens reproduces the rest
        bad = np.where(~ok)[0]; kstar = (bad.max() + 1) if len(bad) else 0
        ent = -(lp.exp() * lp).sum(-1).mean().item()
        kk = max(1, int(0.2 * Lk)); mink = torch.sort(tlp).values[:kk].mean().item()
        out.append(dict(answer_probability=tlp.mean().exp().item(), exact_memorization=float(ok.mean()),
                        extraction_strength=1 - kstar / Lk, min_k_percent=mink, token_entropy=ent))
    return out

@torch.no_grad()
def generate(tok, m, prompts, bs=16, max_new=None):
    outs = []
    for i in range(0, len(prompts), bs):
        enc = [prompt_ids(tok, q) for q in prompts[i:i + bs]]
        w = max(map(len, enc))
        ids = torch.tensor([[tok.pad_token_id] * (w - len(e)) + e for e in enc], device=dev)
        att = torch.tensor([[0] * (w - len(e)) + [1] * len(e) for e in enc], device=dev)
        g = m.generate(input_ids=ids, attention_mask=att, max_new_tokens=max_new or GEN["max_new_tokens"],
                       do_sample=False, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, w:], skip_special_tokens=True)
    return outs

RS = rouge_scorer.RougeScorer(["rougeL"])
def rl(ref, hyp): return RS.score(ref, hyp)["rougeL"].recall

def evaluate(tok, m, df, qcol="question", acol="answer"):
    """Returns dict metric_name -> list of values aligned with df rows (forget metrics),
    plus utility proxy on the fixed retain set."""
    qs, ans = df[qcol].tolist(), df[acol].tolist()
    tf = tf_metrics(tok, m, qs, ans)
    gens = generate(tok, m, qs)
    res = {k: [t[k] for t in tf] for k in tf[0]}
    res["rouge_l"] = [rl(r, g) for r, g in zip(ans, gens)]
    if PT is not None:
        # Truth Ratio (TOFU): mean probability of the wrong answers / probability of the paraphrased correct answer.
        # Lower = the model prefers the correct answer = more knowledge. Direction is handled in the analysis.
        keys = df["original_question"].tolist() if "original_question" in df.columns else qs
        tr = []
        for q, k in zip(qs, keys):
            if k not in PT.index: tr.append(float("nan")); continue
            para = PT.loc[k, "paraphrased_answer"]; perts = list(PT.loc[k, "perturbed_answer"])
            p_para = tf_metrics(tok, m, [q], [para])[0]["answer_probability"]
            p_pert = float(np.mean([x["answer_probability"] for x in tf_metrics(tok, m, [q] * len(perts), perts)]))
            tr.append(p_pert / max(p_para, 1e-12))
        res["truth_ratio"] = tr
    return res

def utility(tok, m):
    t = tf_metrics(tok, m, R.question.tolist(), R.answer.tolist())
    g = generate(tok, m, R.question.tolist())
    return float(np.mean([x["answer_probability"] for x in t])) * 0.5 + \
           float(np.mean([rl(r, h) for r, h in zip(R.answer, g)])) * 0.5   # PROXY, see guide

# ---------- conditions ----------
def lora_relearn(tok, m, n_samples, epochs):
    from peft import LoraConfig, get_peft_model
    seed_all(SEED)
    sub = A.sample(n=n_samples, random_state=SEED)
    pm = get_peft_model(m, LoraConfig(r=L["rank"], lora_alpha=L["alpha"], lora_dropout=L["dropout"],
          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    for n_, p_ in pm.named_parameters():
        if p_.requires_grad: p_.data = p_.data.float()
    opt = torch.optim.AdamW([p_ for p_ in pm.parameters() if p_.requires_grad], lr=L["learning_rate"])
    scaler = torch.cuda.amp.GradScaler(); pm.train()
    data = list(zip(sub.question, sub.answer)); step = 0
    for ep in range(epochs):
        random.Random(SEED + ep).shuffle(data)
        for i in range(0, len(data), L["batch_size"]):
            batch = data[i:i + L["batch_size"]]
            for q, an in batch:
                ids, n = encode_qa(tok, q, an)
                x = torch.tensor([ids], device=dev); lab = x.clone(); lab[:, :n] = -100
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = pm(input_ids=x, labels=lab).loss / (len(batch) * L["gradient_accumulation_steps"])
                scaler.scale(loss).backward()
            step += 1
            if step % L["gradient_accumulation_steps"] == 0 or i + L["batch_size"] >= len(data):
                scaler.step(opt); scaler.update(); opt.zero_grad()
    pm.eval(); return pm.merge_and_unload().eval()

@torch.no_grad()
def rtn_fake_quant(m, bits=4):
    qmax = 2 ** bits - 1
    for name, mod in m.named_modules():
        if isinstance(mod, torch.nn.Linear) and "lm_head" not in name and "embed" not in name:
            w = mod.weight.data.float(); o, i = w.shape
            assert i % GS == 0, f"in_features {i} not divisible by group size {GS}"
            w = w.reshape(o, i // GS, GS)
            if SYM:
                s = (w.abs().amax(-1, keepdim=True) / 2 ** (bits - 1)).clamp(min=1e-8)  # paper [11] Eq.2: max|w| / 2^(N-1)
                q = torch.round(w / s).clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) * s
            else:
                mn, mx = w.amin(-1, keepdim=True), w.amax(-1, keepdim=True)
                s = ((mx - mn) / qmax).clamp(min=1e-8); z = torch.round(-mn / s)
                q = (torch.clamp(torch.round(w / s) + z, 0, qmax) - z) * s
            mod.weight.data = q.reshape(o, i).half()
    return m

def rows_for(cond, strength, ck, meta, df, res, order="", secs=0.0, util=None):
    out = []
    for mn, vals in res.items():
        for qid, au, v in zip(df.question_id, df.author_index, vals):
            out.append(dict(checkpoint_id=ck, method=meta["method"], config_id=meta["config"], question_id=qid,
                author=au, condition=cond, condition_strength=strength, prompt_order=order, metric_name=mn,
                metric_value=v, utility_metric=util, seed=SEED, runtime_seconds=round(secs, 1),
                assigned_member=a.operator, notebook_version=NB_VERSION, config_hash=CFG_HASH))
    return pd.DataFrame(out)

def log(ck, cond, t0, status, err="", reason=""):
    f = "logs/run_log.csv"; new = not os.path.exists(f)
    row = dict(checkpoint_id=ck, condition=cond, start_time=t0.isoformat(), end_time=datetime.datetime.utcnow().isoformat(),
        gpu=gpu, software_version=f"torch{torch.__version__}", notebook_version=NB_VERSION, config_hash=CFG_HASH,
        status=status, error_message=err[:300], rerun_reason=reason, operator=a.operator)
    pd.DataFrame([row]).to_csv(f, mode="a", header=new, index=False)

def run_condition(ck, meta, cond):
    tag = f"{meta['method']}_{int(meta['config']):02d}_{cond}_{a.operator.replace('_','-')}"
    path = f"results/per_question/{tag}.csv"
    if os.path.exists(path): print("skip (done)", tag); return
    t0 = datetime.datetime.utcnow(); s = time.time()
    try:
        tok, m = load(ck)
        strength, order = "", ""
        if cond == "C1-low": m = lora_relearn(tok, m, 20, 1); strength = "20x1"
        if cond == "C1-high": m = lora_relearn(tok, m, 40, 3); strength = "40x3"
        if cond == "C2": m = rtn_fake_quant(m); strength = f"rtn4_g{GS}"
        if cond in ("C0", "C1-low", "C1-high", "C2"):
            res = evaluate(tok, m, B); df = B
            u = utility(tok, m); out = rows_for(cond, strength, ck, meta, df, res, secs=time.time() - s, util=u)
        elif cond == "C3":
            d = Pm.rename(columns={"paraphrased_question": "question", "reference_answer": "answer"})
            res = evaluate(tok, m, d); u = utility(tok, m)
            out = rows_for(cond, "paraphrase", ck, meta, d, res, secs=time.time() - s, util=u)
        elif cond == "C4":
            parts = []
            for order in ["forget_first", "retain_first"]:
                d = M[M.prompt_order == order]
                g = generate(tok, m, d.prompt.tolist(), max_new=2 * GEN["max_new_tokens"])   # v2: two answers need twice the room
                res = {"rouge_l": [rl(r, x) for r, x in zip(d.forget_answer, g)]}
                parts.append(rows_for(cond, "mixed", ck, meta, d, res, order=order, secs=time.time() - s))
            out = pd.concat(parts)
        out.to_csv(path, index=False); log(ck, cond, t0, "ok")
        print(f"{tag}: {time.time()-s:.0f}s")
        del m; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        log(ck, cond, t0, "failed", repr(e)); print("FAILED", tag, repr(e))
        gc.collect(); torch.cuda.empty_cache()

# ---------- which checkpoints ----------
mf = pd.read_csv("manifests/checkpoint_manifest.csv")
jobs = []
if a.controls:
    jobs += [("open-unlearning/tofu_Llama-3.2-1B-Instruct_full", dict(method="ORIGINAL", config=0)),
             ("open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90", dict(method="CONTROL", config=0))]
if a.rows:
    lo, hi = a.rows.split("-") if "-" in a.rows else (a.rows, a.rows)
    for _, r in mf[(mf.checkpoint_number >= int(lo)) & (mf.checkpoint_number <= int(hi))].iterrows():
        jobs.append((r.checkpoint_id, dict(method=r.method, config=r.config_number)))
seed_all(SEED)
for ck, meta in jobs:
    t_ck = time.time()
    for cond in a.conditions.split(","): run_condition(ck, meta, cond)
    print(f"=== {ck.split('/')[-1]} done in {(time.time()-t_ck)/60:.1f} min ===")
    if "unlearn_tofu" in ck:   # v2: free about 2.5 GB per finished checkpoint so the disk does not fill up
        import shutil
        shutil.rmtree(os.path.expanduser("~/.cache/huggingface/hub/models--" + ck.replace("/", "--")), ignore_errors=True)
