#!/bin/bash
set -e

# Default training configuration for remote GPU
# Override any parameter by passing arguments, e.g.:
# ./run_training.sh --epochs 200 --batch_size 256

python train_mnist.py \
    --epochs 100 \
    --batch_size 128 \
    --lr 0.001 \
    --model_base_dim 64 \
    --timesteps 1000 \
    --model_ema_steps 10 \
    --model_ema_decay 0.995 \
    --n_samples 36 \
    "$@"
