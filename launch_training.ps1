# ==============================================================================
# launch_training.ps1 — One-click launcher for Fruit Detection Training
# Handles: Power Plan | Telemetry Agent | HTTP Dashboard Server | Training
# ==============================================================================

$ProjectDir = $PSScriptRoot
$PythonExe  = "$ProjectDir\venv\Scripts\python.exe"
$Port       = 8765

Set-Location $ProjectDir

$env:YOLO_CONFIG_DIR = "$ProjectDir\Ultralytics"
$env:TMP  = "$ProjectDir\tmp"
$env:TEMP = "$ProjectDir\tmp"
New-Item -ItemType Directory -Force -Path $env:TMP | Out-Null

# ==============================================================================
# 1. HIGH PERFORMANCE POWER PLAN
# ==============================================================================
Write-Host ""
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "  [1/4] Activating High Performance Power Plan..." -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan

$hpGuid = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"  # Windows High Performance GUID
try {
    powercfg /setactive $hpGuid
    $activeScheme = powercfg /getactivescheme
    if ($activeScheme -match "High performance") {
        Write-Host "  [OK] High Performance plan is ACTIVE" -ForegroundColor Green
    } else {
        Write-Host "  [OK] Power scheme switched (verify in Control Panel)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  [WARN] Could not set power plan (run as admin for best results)" -ForegroundColor Yellow
}

# ==============================================================================
# 2. KILL ANY STALE PROCESSES (telemetry.py, http server on port)
# ==============================================================================
Write-Host ""
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "  [2/4] Cleaning up stale background processes..." -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan

# Kill any existing telemetry.py processes
Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object {
    $_.MainWindowTitle -eq "" -and $_.CommandLine -like "*telemetry*"
} | Stop-Process -Force -ErrorAction SilentlyContinue

# Kill any existing http.server on our port
$portPid = (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue).OwningProcess
if ($portPid) {
    Stop-Process -Id $portPid -Force -ErrorAction SilentlyContinue
    Write-Host "  [OK] Cleared existing HTTP server on port $Port" -ForegroundColor Yellow
}
Start-Sleep -Milliseconds 500

# ==============================================================================
# 3. START TELEMETRY AGENT (background)
# ==============================================================================
Write-Host ""
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "  [3/4] Starting Hardware Telemetry Agent..." -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan

$telemetryLog = "$ProjectDir\tmp\telemetry.log"
$telemetryProc = Start-Process `
    -FilePath $PythonExe `
    -ArgumentList @("telemetry.py", "--run", "fruit_v5_quality") `
    -WorkingDirectory $ProjectDir `
    -NoNewWindow `
    -RedirectStandardOutput $telemetryLog `
    -RedirectStandardError "$ProjectDir\tmp\telemetry_err.log" `
    -PassThru

Write-Host "  [OK] Telemetry Agent PID: $($telemetryProc.Id)" -ForegroundColor Green
Write-Host "  [OK] Logging to: $telemetryLog" -ForegroundColor Green

# ==============================================================================
# 3b. START HTTP SERVER for Dashboard (background)
# ==============================================================================
Write-Host ""
Write-Host "  Starting HTTP Dashboard Server on port $Port..." -ForegroundColor Cyan

$httpLog = "$ProjectDir\tmp\http_server.log"
$httpProc = Start-Process `
    -FilePath $PythonExe `
    -ArgumentList @("-m", "http.server", "$Port", "--bind", "127.0.0.1") `
    -WorkingDirectory $ProjectDir `
    -NoNewWindow `
    -RedirectStandardOutput $httpLog `
    -RedirectStandardError "$ProjectDir\tmp\http_server_err.log" `
    -PassThru

Start-Sleep -Milliseconds 800

Write-Host "  [OK] HTTP Server PID: $($httpProc.Id)" -ForegroundColor Green
Write-Host ""
Write-Host "  ╔══════════════════════════════════════════════════════╗" -ForegroundColor Magenta
Write-Host "  ║  OPEN DASHBOARD:  http://127.0.0.1:$Port/dashboard.html  ║" -ForegroundColor Magenta
Write-Host "  ╚══════════════════════════════════════════════════════╝" -ForegroundColor Magenta

# Try to open the dashboard in the default browser
try {
    Start-Process "http://127.0.0.1:$Port/dashboard.html"
    Write-Host "  [OK] Dashboard opened in browser!" -ForegroundColor Green
} catch {
    Write-Host "  [INFO] Open manually: http://127.0.0.1:$Port/dashboard.html" -ForegroundColor Yellow
}

# ==============================================================================
# 4. START TRAINING (High priority)
# ==============================================================================
Write-Host ""
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "  [4/4] Launching Training Process (High Priority)..." -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host ""

$trainingArgs = @(
    "train.py",
    "--model", "yolov8s.pt",
    "--data", "data_v5_webcam.yaml",
    "--name", "fruit_v5_quality",
    "--epochs", "120",
    "--batch", "8",
    "--patience", "30",
    "--workers", "0"
)

$trainingProc = Start-Process `
    -FilePath $PythonExe `
    -ArgumentList $trainingArgs `
    -WorkingDirectory $ProjectDir `
    -NoNewWindow `
    -PassThru `
    -Wait:$false

# Set priority immediately
Start-Sleep -Milliseconds 300
try {
    $trainingProc.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::High
    Write-Host "  [OK] Training PID $($trainingProc.Id) set to HIGH priority" -ForegroundColor Green
} catch {
    Write-Host "  [WARN] Could not set priority class (not critical)" -ForegroundColor Yellow
}

# Set telemetry agent to AboveNormal priority
try {
    $telemetryProc.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::AboveNormal
    Write-Host "  [OK] Telemetry agent set to ABOVE NORMAL priority" -ForegroundColor Green
} catch {}

Write-Host ""
Write-Host "===========================================================" -ForegroundColor Green
Write-Host "  ALL SYSTEMS GO" -ForegroundColor Green
Write-Host "  Training PID   : $($trainingProc.Id)  [High]" -ForegroundColor Green
Write-Host "  Telemetry PID  : $($telemetryProc.Id)  [AboveNormal]" -ForegroundColor Green
Write-Host "  HTTP Server PID: $($httpProc.Id)" -ForegroundColor Green
Write-Host "  Dashboard URL  : http://127.0.0.1:$Port/dashboard.html" -ForegroundColor Green
Write-Host "===========================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Waiting for training to complete..." -ForegroundColor Cyan
Write-Host "  Press Ctrl+C to abort - will also close server and telemetry" -ForegroundColor DarkGray
Write-Host ""

# Wait for training to finish
try {
    $trainingProc.WaitForExit()
    $exitCode = $trainingProc.ExitCode
    Write-Host ""
    Write-Host "===========================================================" -ForegroundColor Cyan
    Write-Host "  Training completed with exit code: $exitCode" -ForegroundColor $(if ($exitCode -eq 0) { "Green" } else { "Red" })
    Write-Host "===========================================================" -ForegroundColor Cyan
} finally {
    # Cleanup background processes on exit
    Write-Host "  Shutting down telemetry and HTTP server..." -ForegroundColor Yellow
    Stop-Process -Id $telemetryProc.Id -Force -ErrorAction SilentlyContinue
    Stop-Process -Id $httpProc.Id -Force -ErrorAction SilentlyContinue
    Write-Host "  [OK] Cleanup complete." -ForegroundColor Green
}
