# Ghost Hunter Setup & Run Script for Windows
# Usage: Right-click > Run with PowerShell

Write-Host "👻 Starting Ghost Hunter Setup..." -ForegroundColor Cyan

# 1. Setup Python Virtual Environment
if (-not (Test-Path "venv")) {
    Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

Write-Host "🔌 Activating virtual environment..." -ForegroundColor Green
& .\venv\Scripts\Activate.ps1

Write-Host "⬇️  Installing backend dependencies..." -ForegroundColor Yellow
python -m pip install --upgrade pip
if (Test-Path "requirements_backend.txt") {
    pip install -r requirements_backend.txt
} else {
    pip install -r requirements.txt
}

# 2. Setup Frontend
Write-Host "💻 Checking Frontend dependencies..." -ForegroundColor Cyan
Set-Location ghost-hunter-frontend
if (-not (Test-Path "node_modules")) {
    Write-Host "   Installing node modules..." -ForegroundColor Yellow
    npm install
}
Set-Location ..

# 3. Launch Application
Write-Host "🚀 Launching Ghost Hunter..." -ForegroundColor Magenta
Write-Host "   Backend will open in a separate window." -ForegroundColor Gray
Write-Host "   Frontend will run in this window." -ForegroundColor Gray

# Start Backend in a new window (so logs are visible but don't clutter frontend)
$pythonPath = ".\venv\Scripts\python.exe"
Start-Process -FilePath $pythonPath -ArgumentList "app.py"

# Start Frontend in current window
Write-Host "✨ Starting Frontend..." -ForegroundColor Green
Set-Location ghost-hunter-frontend
npm run dev
