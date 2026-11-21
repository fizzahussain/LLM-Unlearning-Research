# Machine Unlearning in Large Language Models: Distinguishing Genuine Forgetting from Reversible Suppression

Data Analysis and Visualization research project (Phases 1 to 4), FAST-NUCES Islamabad, Department of Data Science.

**Authors:** Hifsa Khattak (24i-2511) and Fizza Hussain (24i-2621). Both authors contributed equally.

- Final paper (Phase 4, IEEE conference format): [`paper/phase4/Phase4_Paper_Hifsa_Fizza.pdf`](paper/phase4/Phase4_Paper_Hifsa_Fizza.pdf), LaTeX source in `paper/phase4/`
- Phase 3 paper: [`paper/Phase3_Paper_Hifsa_Fizza.pdf`](paper/Phase3_Paper_Hifsa_Fizza.pdf)

## Research question

> Under what evaluation conditions can apparent LLM forgetting be reliably distinguished from reversible information suppression?

- **SQ1.** For the same unlearned models, what fraction of models judged "forgotten" recover the knowledge under each condition?
- **SQ2.** How well does each pre-attack metric predict post-attack recovery, measured by Spearman correlation and AUROC?
- **SQ3.** Do these answers change across unlearning methods and hyperparameter configurations?

## What was done

24 released OpenUnlearning checkpoints of Llama-3.2-1B-Instruct unlearned on TOFU forget10 (GradDiff, NPO, SimNPO, RMU; six configurations each) were evaluated before and after five condition settings. Nothing was unlearned by us; the only training is the small LoRA relearning attack.

| Code | Condition | Setting |
|---|---|---|
| C0 | No condition | 200 questions about 10 held-out forget authors (part B) |
| C1-low | Relearning, low budget | LoRA rank 8, 20 examples from 10 other forget authors (part A), 1 epoch |
| C1-high | Relearning, high budget | LoRA rank 8, 40 part-A examples, 3 epochs |
| C2 | Quantization | Simulated 4-bit round-to-nearest, group size 64 |
| C3 | Paraphrased questions | The 200 part-B questions, rewritten |
| C4 | Mixed prompts | Part-B question combined with a retain question, both orders |

Every condition is also applied to the original model and to a retain-only control model that never saw the forget authors.

- Residual knowledge: `K = (S_model - S_control) / (S_original - S_control)`; "K-forgotten" means `K <= 0.10`.
- Control-adjusted recovery: `G = [(S_after - S_before) - (S_control_after - S_control_before)] / (S_original - S_control)`; "recovered" means `G >= 0.50`.

## Main results

- 13 of 24 checkpoints are K-forgotten by answer probability. None of them keeps 90% of utility on TOFU retain questions; four keep it on general-knowledge questions (NPO 3, NPO 5, SimNPO 5, SimNPO 6).
- Of the 13, high-budget relearning overturns the verdict for 4, 5 and 4 over three seeds (33.3% ± 4.4%; seed 0: 4/13, Wilson 95% interval 12.7 to 57.6%). Quantization overturns 1, low-budget relearning and paraphrasing none. Mixed prompts: 1 of 17 by ROUGE-L.
- The four checkpoints that keep general utility are overturned in only 1 of 12 seed runs.
- Standard memorization metrics measured before the attack do not predict relearning recovery (mean Spearman 0.00 to 0.11). Token entropy (-0.73) and Truth Ratio (-0.66) do.
- Recovery is concentrated in RMU and depends strongly on the configuration.
- Recovery on the held-out authors equals recovery on the relearning authors (ablation A5, Spearman 0.99), so it is not memorisation of the relearning examples.
- A rerun reproduced all 22,400 values exactly; scores for the reference models agree with OpenUnlearning's published scores (Pearson 0.99).

These results describe this pool of 24 checkpoints on one benchmark and one model size; intervals are wide. See the paper for limitations.

## Repository structure

```
configs/experiment_config.yaml        settings used by the run script
manifests/checkpoint_manifest.csv     the 24 checkpoints (frozen before any run)
manifests/checkpoint_manifest_selected.csv   how they were selected (pool size and position)
data/processed/                       author split (part A / part B), retain sample, mixed prompts
data/paraphrases/                     200 paraphrased questions, prompt and review record
data/openunlearning_eval/             OpenUnlearning's published evaluation logs for the two reference models (validation)
scripts/run_checkpoints.py            runs all conditions for a checkpoint (needs a GPU)
scripts/analyze_results.py            Phase 3 tables
scripts/analyze_phase4.py             Phase 4 tables (seeds, validation, reproducibility, general utility, question types)
scripts/make_figures.py               Phase 3 figures
scripts/make_figures_phase4.py        figures of the final paper
results/per_question/                 156 result files: 26 models x 6 settings, part B (seed 0)
results/per_question_partA/           78 result files: ablation A5, part A
results/per_question_seed1/, _seed2/  relearning repeated with seeds 1 and 2 (52 files each)
results/per_question_textcheck/       bit-exact rerun of 8 models (reproducibility check)
results/generations/                  saved answers of 8 models, before and after relearning
results/general_utility/              general-knowledge utility (real authors, world facts), 26 models
results/generations_general/          saved answers for the general-utility run
results/analysis/                     tables produced by analyze_results.py
results/analysis_phase4/              tables produced by analyze_phase4.py
figures/                              Phase 3 figures
figures/phase4/                       figures of the final paper (PDF and PNG)
logs/                                 run logs (one row per model and condition)
paper/                                Phase 3 paper and changes from Phase 2
paper/phase4/                         final paper: PDF, LaTeX source and figures
```

## How to reproduce

**Tables and figures (no GPU, a few minutes).** Requires Python with pandas, numpy, scipy and matplotlib.

```
python scripts/analyze_results.py
python scripts/analyze_phase4.py
python scripts/make_figures_phase4.py
```

**Model runs (Kaggle notebook, GPU T4, Internet on, a Hugging Face token saved as the secret `HF_TOKEN`).** About 11 minutes per checkpoint.

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
!git clone -q https://github.com/fizzahussain/LLM-Unlearning-Research.git
%cd LLM-Unlearning-Research
!pip -q install -r requirements.txt
!pip -q uninstall -y torchao
!python -u scripts/run_checkpoints.py --controls --operator member_1      # original and control models
!python -u scripts/run_checkpoints.py --rows 1-24 --operator member_1     # the 24 checkpoints
# ablation A5 (evaluate on the relearning authors):
!python -u scripts/run_checkpoints.py --controls --rows 1-24 --eval-part A --conditions C0,C1-low,C1-high
# relearning seeds 1 and 2:
!python -u scripts/run_checkpoints.py --controls --rows 1-24 --conditions C1-low,C1-high --relearn-seed 1
# general-knowledge utility:
!python -u scripts/run_checkpoints.py --controls --rows 1-24 --general-utility --save-text
```

Use "Save Version > Save & Run All" so that Kaggle keeps the output files.

Software used for the GPU runs: Python 3.13, PyTorch 2.11.0, Transformers 4.57.6, PEFT 0.21.2, Datasets 4.8.5, Accelerate 1.15.0. In total 380 runs, 11.5 GPU-hours on an NVIDIA Tesla T4, none failed.

## Data and model sources

- TOFU dataset: `locuslab/TOFU` on Hugging Face (splits `forget10`, `retain90`, `forget10_perturbed`, `real_authors`, `world_facts`).
- Models: `open-unlearning/tofu_Llama-3.2-1B-Instruct_full` (original), `open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90` (control), and the 24 `open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_*` checkpoints listed in the manifest.
- Framework reference: OpenUnlearning, https://github.com/locuslab/open-unlearning

Model weights and raw datasets are downloaded at run time and are not stored in this repository. Access tokens are never committed.

## Record of changes

`paper/changes_from_phase2.md` lists every difference between the Phase 2 methodology and what was implemented, with the reason for each. Section III-D of the final paper lists the changes and the Phase 4 additions.
