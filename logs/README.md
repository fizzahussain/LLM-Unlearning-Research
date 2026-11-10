# Logs

Every experimental run must be traceable.

## Run log columns

```text
checkpoint_id
condition
start_time
end_time
gpu
software_version
notebook_version
config_hash
status
error_message
rerun_reason
operator
```

Use UTC timestamps inside machine-generated logs and report the local project timezone in the final methodology.

Do not delete failed runs. Mark them clearly and record why a rerun was necessary.

Large raw log files are ignored. Commit the compact CSV run log and any concise failure summary needed to reproduce the experiment.
