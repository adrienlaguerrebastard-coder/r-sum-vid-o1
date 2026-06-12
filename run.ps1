# Script de lancement: ajoute ffmpeg au PATH si besoin, puis demarre l'app
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Cherche ffmpeg dans le PATH, sinon dans l'install winget
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    $ffmpegBin = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Filter "ffmpeg.exe" -Recurse -ErrorAction SilentlyContinue |
                 Select-Object -First 1 -ExpandProperty DirectoryName
    if ($ffmpegBin) {
        $env:PATH = "$ffmpegBin;$env:PATH"
        Write-Host "ffmpeg ajoute au PATH: $ffmpegBin"
    } else {
        Write-Warning "ffmpeg introuvable. Installe-le via: winget install Gyan.FFmpeg"
        exit 1
    }
}

# Cherche tesseract (OCR du score) dans le PATH, sinon dans les emplacements d'install courants
if (-not (Get-Command tesseract -ErrorAction SilentlyContinue)) {
    $tessCandidates = @(
        "$env:LOCALAPPDATA\Programs\Tesseract-OCR",
        "$env:ProgramFiles\Tesseract-OCR",
        "${env:ProgramFiles(x86)}\Tesseract-OCR"
    )
    $tessDir = $tessCandidates | Where-Object { Test-Path (Join-Path $_ "tesseract.exe") } | Select-Object -First 1
    if ($tessDir) {
        $env:PATH = "$tessDir;$env:PATH"
        Write-Host "tesseract ajoute au PATH: $tessDir"
    } else {
        Write-Warning "tesseract introuvable : la lecture du score (buts par OCR) sera desactivee. Installe-le via: winget install UB-Mannheim.TesseractOCR"
    }
}

# venv
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "Creation du venv..."
    python -m venv .venv
    & $python -m pip install --upgrade pip
    & $python -m pip install -r requirements.txt
}

Write-Host ""
Write-Host "======================================" -ForegroundColor Magenta
Write-Host "  TikTok Foot Generator demarre" -ForegroundColor Magenta
Write-Host "  Ouvre http://127.0.0.1:5000" -ForegroundColor Magenta
Write-Host "======================================" -ForegroundColor Magenta
Write-Host ""

& $python app.py
