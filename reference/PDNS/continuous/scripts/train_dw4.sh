#!/bin/bash
#SBATCH --job-name=dw4
#SBATCH --gres=gpu:H200:1
#SBATCH --partition=coe-gpu
#SBATCH --time=16:00:00 # max time (d-hh:mm:ss)
#SBATCH --cpus-per-task=32 # Number of CPU
#SBATCH --output=./sbatch_logs_recent/O-%x.%j
#SBATCH --error=./sbatch_logs_recent/E-%x.%j

srun pwd && SLURM_GPUS_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 python train.py experiment=dw4 \
    resample_batch_size=10000 sigma=3.0 beta0=10.0 beta1=0.1 max_grad_E_norm=100 epsilon=0.01 \
    num_epochs_per_stage=200 num_epochs=10000 eval_freq=1000 save_freq=1000 ema_decay=0.999 \
    optim.lr=0.0001 optim.weight_decay=0.0 optim.beta1=0.5 optim.beta2=0.9 
    # \
    # checkpoint=../../2025-09-21/07-59-32/checkpoints/checkpoint_latest.pt