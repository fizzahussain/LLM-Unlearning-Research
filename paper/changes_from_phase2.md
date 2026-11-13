# Changes from the Phase 2 methodology

Group: Hifsa Khattak (24i-2511), Fizza Hussain (24i-2621)
Last updated: 13 November 2026

This file records every difference between the methodology submitted in Phase 2 and what was implemented in Phase 3, with the reason for each. Items marked TO CONFIRM still need an answer before the paper is written.

## A. Corrections to statements in the Phase 2 document

| # | Phase 2 statement | What we found | Effect |
|---|---|---|---|
| A1 | Step 3: "TOFU releases paraphrased answers, not paraphrased questions." | The TOFU `forget10_perturbed` split contains a `paraphrased_question` column (seen in our run output on 12 Nov 2026). | The statement is withdrawn. Condition C3 still uses our own paraphrases, as designed. TOFU's own paraphrased questions are not used. |
| A2 | Table 6 says the model "fits a free Kaggle T4 or P100 in fp32"; Step 2 says FP16. | The two lines disagreed. | All runs use FP16. |

## B. Implementation choices not fixed in Phase 2

| # | Item | Choice made | Reason |
|---|---|---|---|
| B1 | Checkpoint selection | For each method, all released forget10 Llama-3.2-1B configurations were sorted by the settings in the checkpoint name (learning rate first) and six evenly spaced positions were taken. Pool sizes: GradDiff 40, NPO 54, SimNPO 48, RMU 54. The list was committed to GitHub on 10 Nov 2026, before any model was run. | Phase 2 said "six released configurations per method" without a selection rule. A fixed rule avoids choosing by results. |
| B2 | Author split seed | Seed 0. Part A authors: 1, 4, 5, 6, 7, 8, 12, 13, 15, 19. Part B authors: 0, 2, 3, 9, 10, 11, 14, 16, 17, 18. | Phase 2 said "deterministically"; seed 0 matches the relearning seed. |
| B3 | Retain evaluation sample | A fixed random sample of 200 retain90 questions (seed 0), used for every model. | Phase 2 did not give a size. |
| B4 | LoRA settings | Rank 8 (as submitted); alpha 16, dropout 0.05, learning rate 2e-4, batch size 4, gradient accumulation 4, attention projection layers only. | Phase 2 fixed only the rank and the two budgets. These values are our own choice, not taken from a paper. |
| B5 | Prompt format | System prompt "You are a helpful assistant." and fixed date string "10 Apr 2025", as in the OpenUnlearning model configuration for Llama-3.2-1B-Instruct. | Keeps prompts identical across days and accounts and matches the framework that released the checkpoints. |
| B6 | Mixed prompts (C4) | Each part-B question is paired with one retain question in the form "Answer both questions. 1. ... 2. ...", in both orders. The generation limit is doubled for C4. Only ROUGE-L is measured for C4. | Two answers need more room. Probability-based metrics are not defined for a two-question prompt in our script, so answer probability (the primary metric) is not available for C4. |
| B7 | Quantization (C2) | 4-bit round-to-nearest with the rounding rule of paper [11] (Section 3.1, Eq. 2), applied in groups of 64 weights. | Paper [11] states 4-bit RTN (Section 4.2) and the rounding rule, but no group size. The group size of 64 is our own choice. Phase 2 said the group size would be matched to [11]; that was not possible. |

## C. Items in Phase 2 that were changed or dropped

| # | Phase 2 item | What was done | Reason |
|---|---|---|---|
| C1 | TOFU Model Utility | Replaced by a utility proxy: the mean of answer probability and ROUGE-L on the 200 fixed retain questions. | The full TOFU utility needs three extra evaluation sets. The proxy measures retained knowledge on the same model at low cost. It is not comparable with published Model Utility values. |
| C2 | TOFU Forget Quality | Not computed. | Time; not needed for SQ1 to SQ3. |
| C3 | PCA distance between hidden states | Not computed. | Dropped for time; not needed for SQ1 to SQ3. |
| C4 | OpenUnlearning aggregate score as a baseline | Not computed. | Depends on C2. |
| C5 | Paraphrase failures "are regenerated" | The 20 hand-checked paraphrases: 15 passed, 5 added a word not in the original and were corrected by hand, not regenerated. The other 180 were checked automatically only (no lost names, dates or titles; about 31 add the word "literary" or "literature"). | Recorded in `data/paraphrases/paraphrase_prompt.txt`. |
| C6 | Estimated 30 to 90 minutes per checkpoint | Measured: about 11 minutes per checkpoint on a Kaggle T4 (script version 2). | The full core set of 24 checkpoints was kept. |
| C7 | Ablation A5 (evaluate on part A) | Run on 13 November 2026 as a separate run (26 models x C0, C1-low, C1-high on the 200 part-A questions; 78 runs, all ok), using the option `--eval-part A` added to the run script (version 3). The part-B procedure is unchanged. | The first versions of the run script evaluated part B only. |
| C8 | Recovery curves over relearning steps | Not produced. | Only the final relearned model is evaluated. |
| C9 | Reading generated outputs by question type (Step 11) | Not done. | The run script saved scores, not generated text. |
| C10 | Verdict-flip rate | No checkpoint met the full verdict (K <= 0.10 and utility >= 90%), so flip rates are reported for the 13 checkpoints with K <= 0.10 only. | This is the Phase 2 definition of the flip rate (Table 7). |
| C14 | Figures | Colour added (colour-blind-safe palette plus marker shapes). | Readability. |
| C12 | Heatmap of AUROC | Heatmap of Spearman correlation. | Checkpoint-level AUROC could not be computed (4 recovered checkpoints, below the pre-set minimum of 5). |
| C13 | Over-refusal analysis on retain questions | Not done. | Generated text was not saved. |
| C11 | Truth Ratio aggregation | Median over questions instead of mean. | The per-question ratio has extreme values. |

## D. Run history

| Date | What | Script version | Used in the paper? |
|---|---|---|---|
| 11 Nov 2026 | Controls and rows 1 to 12 (Fizza) | v1 (no system prompt, no Truth Ratio) | No. Discarded. |
| 12 Nov 2026 | Controls and rows 1 to 12 rerun (Fizza), 84 runs, all ok | v2 | Yes |
| 12 Nov 2026 | Rows 13 to 24 (Hifsa), interactive session; files lost when the session ended | v2 | No |
| 13 Nov 2026 | Rows 13 to 24 rerun (Hifsa), background version, 72 runs, all ok | v2 | Yes |
| 13 Nov 2026 | Ablation A5: all 26 models on part A (Hifsa), 78 runs, all ok | v3 (adds `--eval-part`; same configuration hash) | Yes (Section 7, Figure 9) |

## E. Feedback from class

Written feedback on Phase 2 had not been received by 13 November 2026. The Phase 1 feedback (no names; no reference list or venues; mismatch between paper [10] in the summary and row 10 of the table; gaps generic and lightly evidenced) was addressed in Phase 2 and carried into Phase 3; see Table 2 of the paper.
