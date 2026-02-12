#!/bin/bash
# This is a setup script that is specifically designed to work on 
# the Milwaukee School of Engineering's (MSOE) supercomputer, ROSIE.
# The configurations, module loads, partitions, and environment setup are 
# tailored for ROSIE's infrastructure and may not work elsewhere without modification.

PARTITION=${1:-"teaching"}

AVAILABLE_PARTITIONS=(dgxh100 dgx teaching)

if ! [[ " ${AVAILABLE_PARTITIONS[@]} " =~ " ${PARTITION} " ]]; then
    echo "Invalid partition: $PARTITION"
    echo "Available partitions: ${AVAILABLE_PARTITIONS[@]}"
    exit 1
fi

GPU_COUNT=${2:-1}

if ! [[ $GPU_COUNT -ge 1 && $GPU_COUNT -le 8 ]]; then
    echo "Invalid GPU count: $GPU_COUNT"
    echo "GPU count must be between 1 and 8"
    exit 1
fi

source /etc/profile.d/lmod.sh
module load cuda/12.9

# CUDA/NCCL from venv (uv/pip install nvidia-*)
VENV_LIB=".venv/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="${VENV_LIB}/cuda_runtime/lib:${VENV_LIB}/nccl/lib:$LD_LIBRARY_PATH"
export GRADTUITY_NCCL_LIBRARY="libnccl.so.2"
export GRADTUITY_LIBCUDART_LIBRARY="libcudart.so"

srun --partition=$PARTITION --time=08:00:00 --gres=gpu:$GPU_COUNT --cpus-per-task=32 --mem=200G --account=undergrad_research --pty bash