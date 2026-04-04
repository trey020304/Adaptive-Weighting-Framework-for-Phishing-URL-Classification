#!/bin/bash
# ============================================================
# GPU Runner for WSL2 / Linux
# ============================================================
# Sets up CUDA library paths from pip-installed nvidia packages
# and runs any pipeline script with GPU support.
#
# Usage:
#   ./run_gpu.sh <script.py> [args...]
#
# Examples:
#   ./run_gpu.sh adaptive_hybrid.py
#   ./run_gpu.sh bigru_pipeline.py 7
#   ./run_gpu.sh odae_wpdc_pipeline.py
#   ./run_gpu.sh pso_xgboost_pipeline.py 3
#   ./run_gpu.sh path_b_baseline.py
#   ./run_gpu.sh inference.py --checkpoint results/adaptive_hybrid/dataset_3/checkpoint --dataset_id 10
# ============================================================

set -e

# Resolve project root (directory containing this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/thesis/.venv/bin/activate" ]; then
    source "$HOME/thesis/.venv/bin/activate"
else
    echo "ERROR: Could not find virtual environment (.venv/bin/activate)"
    exit 1
fi

# Build LD_LIBRARY_PATH from pip-installed NVIDIA packages
SITE="$(python -c 'import site; print(site.getsitepackages()[0])')/nvidia"
if [ -d "$SITE" ]; then
    NVIDIA_LIBS=""
    for lib_dir in "$SITE"/*/lib; do
        [ -d "$lib_dir" ] && NVIDIA_LIBS="${NVIDIA_LIBS:+$NVIDIA_LIBS:}$lib_dir"
    done
    export LD_LIBRARY_PATH="${NVIDIA_LIBS}:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
    echo "CUDA libraries loaded from pip packages"
else
    echo "WARNING: No pip nvidia packages found at $SITE"
    echo "Install with: pip install 'tensorflow[and-cuda]'"
fi

echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo ""

if [ $# -eq 0 ]; then
    echo "Usage: ./run_gpu.sh <script.py> [args...]"
    echo ""
    echo "Available pipelines:"
    echo "  adaptive_hybrid.py          Adaptive hybrid (Path A + Path B)"
    echo "  path_a_baseline.py          Chi-square + Random Forest"
    echo "  path_b_baseline.py          CNN-BiLSTM (GPU-accelerated)"
    echo "  bigru_pipeline.py [id]      BiGRU-Attention (GPU-accelerated)"
    echo "  odae_wpdc_pipeline.py [id]  ODAE-WPDC autoencoder (GPU-accelerated)"
    echo "  pso_xgboost_pipeline.py [id] PSO-XGBoost"
    echo "  inference.py --checkpoint ... --dataset_id ..."
    exit 0
fi

exec python "$@"
