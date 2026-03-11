## Workflow

This repository uses a protected Git workflow.

- Do not commit directly to `main`.
- All changes must be made on a feature branch and merged through a Pull Request.
- Each Pull Request requires at least 1 approving review before merge.
- Direct pushes to `main` are blocked by repository rules.

### Branch naming

Use descriptive branch names, for example:

- `feature/add-inference-buffer`
- `fix/websocket-reconnect`
- `docs/update-readme`
- `chore/setup-ci`

### Pull Requests

Before opening a Pull Request:

- Make sure your branch is up to date.
- Describe the scope of changes clearly.
- Link the related GitHub issue if applicable.

Pull Requests are merged into `main` using **Squash** to keep history clean.
