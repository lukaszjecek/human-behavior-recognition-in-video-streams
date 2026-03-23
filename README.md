## Run dev environment    
    
### Requirements:    
- Docker  
- local clone of this repository    
- dataset subset in `./data/raw`   

### Setup:  
```
docker compose up --build
```  
  
### Dataset layout  
  
Place the extracted dataset subset under:  
  
- `./data/raw`  
  
The application scans this directory recursively for `.mp4` files, so nested class folders are supported.  
  
### Expected output  
  
After running:  
  
```
docker compose up --build
```  
  
The container should start successfully, print a short dataset summary in logs, and create:  
  
- `./data/logs/startup_summary.json`  
  
If the dataset subset is mounted correctly, the logs should include the number of discovered video files and classes.  
  
### Windows note  
  
On Windows PowerShell, create the folders manually if they do not already exist:  
  
```
mkdir data\raw  
mkdir data\logs  
docker compose up --build  
```

 ### Workflow    
 This repository uses a protected Git workflow.    
    
- Do not commit directly to `main`.    
- All changes must be made on a feature branch and merged through a Pull Request.    
- Each Pull Request requires at least 1 approving review before merge.    
- Direct pushes to `main` are blocked by repository rules.
- Repository ownership is defined in `.github/CODEOWNERS`.
- When a Pull Request changes files owned by specific contributors, GitHub automatically requests their review once the PR is ready for review.
- Changes in owned areas should be approved by the relevant code owner before merge. Such approval is mandatory for merging.
    
### Branch naming
Create a feature branch from the related GitHub Issue page.
Use the branch name suggested by GitHub when creating the branch from the issue, unless there is a strong reason to adjust it.

### Pull Requests    
 Before opening a Pull Request:    
    
- Make sure your branch is up to date.    
- Describe the scope of changes clearly.    
- Link the related GitHub issue if applicable.    
    
Pull Requests are merged into `main` using **Squash** to keep history clean.


### Dev commands

Run the development container:
```
.\scripts\run.ps1
```

Run tests:
```
.\scripts\test.ps1
```

Run lint checks:
```
.\scripts\lint.ps1
```


> If PowerShell blocks scripts from running, set this locally:
> ```
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```


### Run CI checks locally

```
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
python -m ruff check .
python -m pytest -q tests
```