# ============================================================
# GPU Runner for Windows (via WSL2)
# ============================================================
# Launches a pipeline script inside WSL2 with GPU support.
# TensorFlow on native Windows has no GPU support (post TF 2.10),
# so this script delegates to WSL2 where CUDA works.
#
# Prerequisites:
#   1. WSL2 installed with a Linux distro (e.g. Ubuntu)
#   2. Inside WSL, create a venv and install tensorflow[and-cuda]:
#        cd /mnt/c/Users/ASUS/Desktop/"THESIS II"
#        python3 -m venv .venv-wsl
#        source .venv-wsl/bin/activate
#        pip install 'tensorflow[and-cuda]' scikit-learn pandas xgboost pyyaml joblib
#   3. NVIDIA GPU drivers installed on Windows (nvidia-smi should work)
#
# Usage (from PowerShell):
#   .\run_gpu.ps1 <script.py> [args...]
#
# Examples:
#   .\run_gpu.ps1 adaptive_hybrid.py
#   .\run_gpu.ps1 bigru_pipeline.py 7
#   .\run_gpu.ps1 odae_wpdc_pipeline.py
#   .\run_gpu.ps1 pso_xgboost_pipeline.py 3
#   .\run_gpu.ps1 inference.py --checkpoint results/adaptive_hybrid/dataset_3/checkpoint --dataset_id 10
# ============================================================

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs
)

if (-not $ScriptArgs -or $ScriptArgs.Count -eq 0) {
    Write-Host "Usage: .\run_gpu.ps1 <script.py> [args...]" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Available pipelines:"
    Write-Host "  adaptive_hybrid.py           Adaptive hybrid (Path A + Path B)"
    Write-Host "  path_a_baseline.py           Chi-square + Random Forest"
    Write-Host "  path_b_baseline.py           CNN-BiLSTM (GPU-accelerated)"
    Write-Host "  bigru_pipeline.py [id]       BiGRU-Attention (GPU-accelerated)"
    Write-Host "  odae_wpdc_pipeline.py [id]   ODAE-WPDC autoencoder (GPU-accelerated)"
    Write-Host "  pso_xgboost_pipeline.py [id] PSO-XGBoost"
    Write-Host "  inference.py --checkpoint ... --dataset_id ..."
    exit 0
}

# Convert Windows path to WSL path
$WinDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$WslDir = wsl wslpath -u "$WinDir" 2>$null
if (-not $WslDir) {
    $WslDir = "/mnt/c/Users/ASUS/Desktop/THESIS II"
}

$ArgsString = ($ScriptArgs | ForEach-Object {
    if ($_ -match '\s') { "'$_'" } else { $_ }
}) -join " "

Write-Host "Running on WSL2 with GPU support..." -ForegroundColor Green
Write-Host "  Directory: $WslDir" -ForegroundColor DarkGray
Write-Host "  Command:   ./run_gpu.sh $ArgsString" -ForegroundColor DarkGray
Write-Host ""

wsl bash -c "cd '$WslDir' && chmod +x run_gpu.sh && ./run_gpu.sh $ArgsString"
