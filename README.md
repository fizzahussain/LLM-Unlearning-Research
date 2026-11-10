# LLM Unlearning Research

## Distinguishing Genuine Forgetting from Reversible Suppression

This repository contains the implementation and experimental evaluation for a Data Analysis and Visualization research project on large language model unlearning.

The study investigates whether models that appear to have forgotten selected information have genuinely removed that information or have only suppressed access to it under standard evaluation.

## Research question

> When an unlearned language model appears to forget information, does that information remain inaccessible under relearning, quantization, paraphrased questions, and mixed-prompt evaluation?

The project evaluates whether common unlearning metrics predict later recovery and whether different recovery conditions produce consistent conclusions.

## Core experimental scope

- **Dataset:** TOFU forget10 and retain90
- **Architecture:** Llama-3.2-1B-Instruct
- **Unlearning methods:** GradDiff, NPO, SimNPO, and RMU
- **Released configurations:** 6 per method
- **Core checkpoints:** 24
- **Forget-set split:** 10 authors for relearning and 10 unseen authors for evaluation
- **Recovery conditions:** LoRA relearning, fake 4-bit quantization, paraphrased questions, and mixed prompts
- **Primary analysis:** normalized residual knowledge, recovery gain, utility retention, and verdict-flip rate

The unlearned checkpoints are evaluated rather than trained from scratch. Training is limited to small LoRA relearning attacks used to test recoverability.

## Data and model sources

- [TOFU dataset](https://huggingface.co/datasets/locuslab/TOFU)
- [OpenUnlearning repository](https://github.com/locuslab/open-unlearning)
- [OpenUnlearning TOFU checkpoint collection](https://huggingface.co/collections/open-unlearning/tofu-unlearned-models)
- [OpenUnlearning Hugging Face organization](https://huggingface.co/open-unlearning)

Raw datasets and model weights are downloaded at runtime and are not committed to this repository.

## Experimental design

### Data split

The TOFU forget10 split contains 20 fictitious authors and 400 question-answer pairs. Authors are divided before model evaluation:

- **Part A:** 10 authors and 200 questions for relearning
- **Part B:** 10 different authors and 200 questions for recovery evaluation

The author split is frozen in a versioned manifest. Questions from one author must never appear in both parts.

### Experimental conditions

| Code | Condition | Purpose |
|---|---|---|
| C0 | Original prompt evaluation | Measure pre-attack forgetting |
| C1-Low | LoRA relearning using 20 samples for 1 epoch | Test low-budget recovery |
| C1-High | LoRA relearning using 40 samples for 3 epochs | Test stronger recovery |
| C2 | Fake 4-bit round-to-nearest quantization | Test recovery caused by weight transformation |
| C3 | Paraphrased Part B questions | Test prompt-surface robustness |
| C4 | Mixed forget and retain prompts in both orders | Test contextual recovery and prompt-order effects |

### Main measures

Normalized residual knowledge is calculated for every metric and condition:

```text
K = (S_model - S_control) / (S_original - S_control)
```

A checkpoint is treated as forgotten under the primary rule when:

```text
K <= 0.10 and utility retention >= 0.90
```

Control-adjusted recovery gain is:

```text
G = ((S_after - S_before) - (S_control_after - S_control_before))
    / (S_original - S_control)
```

The primary recovery threshold is `G >= 0.50`. Sensitivity analysis also uses 0.25 and 0.75.

Denominators are computed separately for every metric and experimental condition.

## Repository structure

```text
.
├── .github/
│   └── pull_request_template.md
├── configs/
│   └── experiment_config.yaml
├── data/
│   └── README.md
├── figures/
│   └── README.md
├── logs/
│   └── README.md
├── manifests/
│   └── checkpoint_manifest.csv
├── notebooks/
│   └── README.md
├── results/
│   └── README.md
├── .gitignore
├── CONTRIBUTING.md
├── README.md
└── requirements.txt
```

## Planned notebooks

1. **Environment and data validation**
2. **Author split and prompt preparation**
3. **Control and baseline evaluation**
4. **Checkpoint recovery experiments**
5. **Metric calculation and statistical analysis**
6. **Ablations and visualizations**

Notebook responsibilities and expected outputs are documented in [notebooks/README.md](notebooks/README.md).

## Two-member workflow

The 24 checkpoints remain one experiment but are divided evenly for execution:

| Member | Assigned methods | Checkpoints |
|---|---|---:|
| Member 1 | GradDiff and NPO | 12 |
| Member 2 | SimNPO and RMU | 12 |

Both members must use the same frozen manifests, prompts, configuration, metric definitions, and output schema. Every result row records the operator, configuration, notebook version, and checkpoint identifier.

See [CONTRIBUTING.md](CONTRIBUTING.md) for branch, review, and merge rules.

## Reproducibility rules

1. Freeze the checkpoint manifest before running recovery experiments
2. Select checkpoints without examining recovery outcomes
3. Use an author-level split rather than a question-level random split
4. Store generated paraphrases and their quality-review status
5. Write result rows immediately after each condition
6. Preserve per-question results before producing summaries
7. Record software versions, GPU type, seed, runtime, and failures
8. Never commit access tokens, model weights, raw caches, or private credentials
9. Report unavailable checkpoints and reruns rather than silently replacing results
10. State conclusions as applying under the tested conditions, not as proof of permanent deletion

## Project status

- [x] Research problem and literature review
- [x] Phase 2 methodology design
- [x] Repository initialization
- [ ] Freeze 24-checkpoint manifest
- [ ] Validate TOFU data
- [ ] Freeze author-level split
- [ ] Prepare paraphrased and mixed prompts
- [ ] Complete one end-to-end pilot
- [ ] Run 24-checkpoint experiment
- [ ] Calculate baselines and ablations
- [ ] Produce final figures
- [ ] Complete Phase 3 report

## Course information

**Course:** Data Analysis and Visualization  
**Project:** LLM Unlearning: Distinguishing Genuine Forgetting from Reversible Suppression

## Citation and responsible use

This repository is intended for reproducible academic evaluation. Cite TOFU, OpenUnlearning, the evaluated unlearning methods, and all comparative studies in the final paper. Follow the licenses and access requirements of the original datasets and model checkpoints.
