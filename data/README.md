# Data

## Runtime sources

The project downloads data from the official TOFU dataset:

- `locuslab/TOFU`
- `forget10`
- `retain90`
- required utility-evaluation splits

Raw downloaded files belong in `data/raw/` and are ignored by Git.

## Versioned research assets

The following generated assets should be committed because they define the experiment:

- author split manifest
- Part A and Part B row identifiers
- fixed retain evaluation identifiers
- paraphrased Part B questions
- paraphrase quality-review decisions
- mixed-prompt definitions
- data validation summary

Do not commit access tokens, Hugging Face caches, model weights, or unrelated personal data.

## Required validation

Before experimentation, verify:

- 400 forget10 rows
- 20 forget10 authors
- 20 questions per forget author
- 3,600 retain90 rows
- no duplicate record identifiers
- no missing questions or answers
- no author overlap between Part A and Part B

The frozen split uses 10 authors for Part A relearning and 10 different authors for Part B evaluation.
