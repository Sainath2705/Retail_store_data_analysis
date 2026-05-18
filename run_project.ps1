$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $ProjectRoot ".venv312\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    Write-Host "Project Python was not found at .venv312\Scripts\python.exe" -ForegroundColor Yellow
    Write-Host "Create it with:" -ForegroundColor Yellow
    Write-Host "  C:\Users\SAINATH\AppData\Local\Programs\Python\Python312\python.exe -m venv .venv312"
    Write-Host "  .\.venv312\Scripts\python.exe -m pip install -r requirements.txt"
    exit 1
}

& $PythonExe (Join-Path $ProjectRoot "run.py")
