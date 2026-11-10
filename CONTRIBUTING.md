# Contributing

## Working agreement

This is a two-member research project. The core experiment contains 24 checkpoints and must remain one consistent study even though execution is divided between members.

## Work allocation

| Member | Methods | Checkpoints |
|---|---|---:|
| Member 1 | GradDiff and NPO | 12 |
| Member 2 | SimNPO and RMU | 12 |

Both members run every required condition for their assigned checkpoints.

## Branch workflow

Create a separate branch for each unit of work:

- `setup/<short-name>`
- `data/<short-name>`
- `experiment/<method-name>`
- `analysis/<short-name>`
- `figures/<short-name>`
- `docs/<short-name>`

Keep commits focused and use clear messages. Open a pull request into `main` and have the other member review it before merging.

## Files that must be frozen before the full run

- `configs/experiment_config.yaml`
- `manifests/checkpoint_manifest.csv`
- author split manifest
- Part A and Part B identifiers
- paraphrase file
- mixed-prompt file
- result schemas
- package versions

If a frozen file changes, document the reason and identify which experiments must be rerun.

## Experiment rules

1. Run one end-to-end pilot before all checkpoints
2. Process one model checkpoint at a time
3. Save after every experimental condition
4. Keep question-level data
5. Use the same prompts, seeds, metrics, and thresholds
6. Record failed and repeated runs
7. Do not select checkpoints after inspecting recovery results
8. Do not commit model weights or access tokens
9. Do not overwrite another member's result file
10. Review two completed checkpoints from the other member

## Result file naming

Use:

```text
<method>_<config-number>_<condition>_<operator>.csv
```

Example:

```text
NPO_03_C1-high_member-1.csv
```

## Pull request checklist

Every experimental pull request should state:

- checkpoint IDs
- conditions completed
- notebook or script version
- GPU type
- runtime
- output files
- missing or failed runs
- validation checks performed
