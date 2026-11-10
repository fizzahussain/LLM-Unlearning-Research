# Notebooks

Run notebooks in numerical order. Each notebook must save its outputs before the GPU session ends.

## 01 Environment and data validation

Responsibilities:

- install or verify packages
- authenticate with Hugging Face secrets
- record GPU and software information
- load TOFU
- validate expected row and author counts

Outputs:

- environment report
- data validation report
- raw split metadata

## 02 Author split and prompt preparation

Responsibilities:

- create the seeded author-level split
- freeze Part A and Part B identifiers
- prepare retain evaluation identifiers
- prepare paraphrase and mixed-prompt assets
- perform paraphrase quality review

Outputs:

- author split manifest
- Part A and Part B files
- paraphrase file
- mixed-prompt file

## 03 Control and baseline evaluation

Responsibilities:

- evaluate the original full model
- evaluate the retain90 control
- compute condition-specific control spans
- validate metric directions

Outputs:

- original model results
- control model results
- denominator validation table

## 04 Checkpoint recovery experiments

Responsibilities:

- load one released checkpoint at a time
- evaluate C0
- run C1-low and C1-high
- run C2
- evaluate C3 and C4
- save question-level and checkpoint-level results
- clear model files before continuing

Outputs:

- per-condition result files
- runtime logs
- failure and rerun records

## 05 Metrics and statistical analysis

Responsibilities:

- calculate normalized residual knowledge K
- calculate control-adjusted recovery gain G
- calculate utility retention
- assign forgetting and recovery verdicts
- compute verdict-flip rates
- run correlations, AUROC checks, and descriptive uncertainty analyses

Outputs:

- merged checkpoint summary
- baseline comparison table
- statistical results

## 06 Ablations and visualizations

Responsibilities:

- run offline ablations
- create publication-ready figures
- produce tables for the Phase 3 report
- record plotting data

Outputs:

- ablation results
- final figures
- figure captions
- comparison tables
