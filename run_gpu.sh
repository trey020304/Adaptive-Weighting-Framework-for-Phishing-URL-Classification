#!/bin/bash
source ~/thesis/.venv/bin/activate

SITE=~/thesis/.venv/lib/python3.13/site-packages/nvidia
export LD_LIBRARY_PATH="$SITE/cublas/lib:$SITE/cuda_runtime/lib:$SITE/cudnn/lib:$SITE/cufft/lib:$SITE/cusparse/lib:$SITE/cusolver/lib:$SITE/nvjitlink/lib:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"

echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

cd ~/thesis
exec python "$@"
