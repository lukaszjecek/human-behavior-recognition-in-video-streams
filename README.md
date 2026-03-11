## Run dev environment    
    
### Requirements:    
- Docker  
- local clone of this repository    
- dataset subset in `./data/subset`   

### Setup:  
```
docker compose up --build
```  
  
### Dataset layout  
  
Place the extracted dataset subset under:  
  
- `./data/subset`  
  
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
mkdir data\subset  
mkdir data\logs  
docker compose up --build  
```

 ### Workflow    
 This repository uses a protected Git workflow.    
    
- Do not commit directly to `main`.    
- All changes must be made on a feature branch and merged through a Pull Request.    
- Each Pull Request requires at least 1 approving review before merge.    
- Direct pushes to `main` are blocked by repository rules.    
    
### Branch naming    
 Use descriptive branch names, for example:    
    
- `feature/add-inference-buffer` - `fix/websocket-reconnect` - `docs/update-readme` - `chore/setup-ci`    
 ### Pull Requests    
 Before opening a Pull Request:    
    
- Make sure your branch is up to date.    
- Describe the scope of changes clearly.    
- Link the related GitHub issue if applicable.    
    
Pull Requests are merged into `main` using **Squash** to keep history clean.