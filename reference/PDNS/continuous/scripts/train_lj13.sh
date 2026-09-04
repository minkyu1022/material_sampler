#!/bin/bash
#SBATCH --job-name=lj13
#SBATCH --gres=gpu:H100:1
#SBATCH --partition=coe-gpu
#SBATCH --time=16:00:00 # max time (d-hh:mm:ss)
#SBATCH --cpus-per-task=32 # Number of CPU
#SBATCH --output=./sbatch_logs_recent/O-%x.%j
#SBATCH --error=./sbatch_logs_recent/E-%x.%j

# srun pwd && SLURM_GPUS_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 python train.py experiment=ad

# srun pwd && SLURM_GPUS_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 python train.py experiment=lj55 resample_batch_size=10000
# srun pwd && SLURM_GPUS_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 python train.py experiment=lj13 \
#     resample_batch_size=10000 \
#     sigma=2.0 \
#     beta1=0.1 \
#     max_grad_E_norm=100 \
#     epsilon=1.0 \
#     num_epochs_per_stage=1000 \
#     num_epochs=50000 \
#     eval_freq=1000 save_freq=1000 \
#     ema_decay=0.999 seed=1
srun pwd && SLURM_GPUS_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 python train.py experiment=lj13 \
    resample_batch_size=10000 sigma=2.0 beta1=0.1 max_grad_E_norm=100 fix_gamma=0.1 epsilon=1000000 num_epochs_per_stage=200 num_epochs=50000 eval_freq=1000 save_freq=200 ema_decay=0.999