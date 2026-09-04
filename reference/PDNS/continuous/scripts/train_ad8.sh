#!/bin/bash
#SBATCH --job-name=ad2
#SBATCH --gres=gpu:H200:1
#SBATCH --partition=coe-gpu
#SBATCH --time=15:00:00 # max time (d-hh:mm:ss)
#SBATCH --cpus-per-task=32 # Number of CPU
#SBATCH --output=./sbatch_logs/O-%x.%j
#SBATCH --error=./sbatch_logs_recent/E-%x.%j

# srun pwd && SLURM_GPUS_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 python train.py experiment=ad_ic \
#     num_epochs_per_stage=1000 num_epochs=100000 \
#     save_freq=1000 epsilon=1.0 

srun pwd && SLURM_GPUS_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 python train.py \
    experiment=ad_ic save_freq=1000 \
    epsilon=0.01 ema_decay=0.999 \
    optim.beta1=0.0 optim.beta2=0.999 \
    num_epochs_per_stage=100 num_epochs=1000000 \
    optim.lr=1e-4 resample_batch_size=10000 \
    annealing_type=annealed max_grad_E_norm=100.0 \
    beta0=10 beta1=0.1 annealing_beta1=0.1 \
    sigma=2.0 clip_grad_norm=10.0

# srun pwd && SLURM_GPUS_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 python train.py experiment=ad_ic save_freq=1000 \
#     epsilon=0.02 ema_decay=0.999 optim.beta1=0.0 optim.beta2=0.999 \
#     num_epochs_per_stage=1000 num_epochs=1000000 \
#     optim.lr=1e-5 resample_batch_size=10000 \
#     annealing_type=annealed max_grad_E_norm=100.0 \
#     score_matcher.buffer_size=100000 \
#     beta1=0.001 annealing_beta1=0.001

# srun pwd && SLURM_GPUS_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 python train.py experiment=ad num_epochs_per_stage=10000
