# CI Workflows

[Back to README](../README.md)

## CI

- Trigger: `pull_request` to `main`, `workflow_dispatch`
- Purpose: run Python tests and Ruff linting
- Behavior:
  - on pull requests, Ruff checks changed Python files
  - on manual runs, Ruff checks all Python files under `src` and `tests`

## Docker Smoke

- Trigger: `pull_request` to `main`, `workflow_dispatch`
- Purpose: verify that Docker Compose can build and run the inference service on a tiny MP4 sample
- Behavior:
  - generates a deterministic sample of 3 MP4 files
  - runs the inference service through Docker Compose
  - verifies that `data/logs/startup_summary.json` is created
  - checks that the summary reports 3 videos and 3 classes