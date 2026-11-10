# Results

Commit compact CSV outputs and analysis summaries. Do not commit model weights or large caches.

## Question-level schema

Store question-level files in `results/per_question/`.

Required columns:

```text
checkpoint_id
method
config_id
question_id
author
condition
condition_strength
prompt_order
metric_name
metric_value
utility_metric
seed
runtime_seconds
assigned_member
notebook_version
config_hash
```

## Checkpoint-level schema

Store summaries in `results/per_checkpoint/`.

Required columns:

```text
checkpoint_id
method
config_id
condition
condition_strength
metric_name
score_before
score_after
original_score
control_score
K
G
utility_before
utility_after
forgotten_before
recovered_after
verdict_flip
```

## Ablation schema

Store ablation outputs in `results/ablations/`.

Each row should identify:

- ablation name
- changed component
- reference setting
- ablated setting
- metric
- reference result
- ablated result
- difference
- interpretation

## Validation before merging

- no duplicate checkpoint-condition-question-metric rows
- all checkpoint IDs occur in the frozen manifest
- all numeric metrics parse correctly
- every row identifies its operator
- failed runs are absent from final summaries but retained in the run log
- results contain no access tokens or local cache paths
