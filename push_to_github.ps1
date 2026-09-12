# Pushes the prepared analysis pipeline, affiliation analysis and test suite to
# https://github.com/CHEVVA181/Crypto-Investment-Bias-in-LLMs
#
# Run it from PowerShell, in this folder:
#     powershell -ExecutionPolicy Bypass -File .\push_to_github.ps1
#
# It clones the repo into a temp folder, copies these files in, commits and
# pushes. Nothing in this folder is modified.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$repo = "https://github.com/CHEVVA181/Crypto-Investment-Bias-in-LLMs.git"
$work = Join-Path $env:TEMP ("cib_push_" + [guid]::NewGuid().ToString("N").Substring(0, 8))

Write-Host "Cloning $repo ..." -ForegroundColor Cyan
git clone --quiet $repo $work
if ($LASTEXITCODE -ne 0) { throw "git clone failed" }

Write-Host "Copying files ..." -ForegroundColor Cyan
Copy-Item -Path "README.md", "requirements.txt", ".gitignore" -Destination $work -Force
New-Item -ItemType Directory -Force -Path (Join-Path $work "src"), (Join-Path $work "tests") | Out-Null
Copy-Item -Path "src\*.py"   -Destination (Join-Path $work "src")   -Force
Copy-Item -Path "tests\*.py" -Destination (Join-Path $work "tests") -Force

Push-Location $work
try {
    git add -A
    if ((git status --porcelain).Length -eq 0) {
        Write-Host "Nothing to push - the repo already matches these files." -ForegroundColor Yellow
        return
    }
    git commit --quiet -m "Add analysis pipeline, affiliation analysis and test suite"
    if ($LASTEXITCODE -ne 0) { throw "git commit failed" }

    Write-Host "Pushing to main ..." -ForegroundColor Cyan
    git push origin main
    if ($LASTEXITCODE -ne 0) { throw "git push failed - check your GitHub credentials" }

    Write-Host ""
    Write-Host "Done: https://github.com/CHEVVA181/Crypto-Investment-Bias-in-LLMs" -ForegroundColor Green
}
finally {
    Pop-Location
}
