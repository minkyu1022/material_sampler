#!/bin/bash
#SBATCH --job-name=lj55
#SBATCH --gres=gpu:H200:1
#SBATCH --partition=coe-gpu
#SBATCH --time=16:00:00 # max time (d-hh:mm:ss)
#SBATCH --cpus-per-task=32 # Number of CPU
#SBATCH --output=./sbatch_logs_recent/O-%x.%j
#SBATCH --error=./sbatch_logs_recent/E-%x.%j

# srun pwd && SLURM_GPUS_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 python train.py experiment=ad

srun pwd && SLURM_GPUS_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 python train.py experiment=lj55 \
    resample_batch_size=10000 num_epochs_per_stage=500 \
    beta1=0.1 \
    max_grad_E_norm=1000 annealing_beta0=1.0 annealing_beta1=0.001 annealing_type='langevin' \
    save_freq=500 seed=0 \
    checkpoint=../../2025-09-24/16-02-09/checkpoints/checkpoint_latest.pt
    # quick_evaluation=false
# srun pwd && SLURM_GPUS_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 python train.py experiment=lj13  resample_batch_size=10000

# srun pwd && SLURM_GPUS_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 python train.py experiment=lj55 \
#     resample_batch_size=10000 num_epochs_per_stage=1000 beta1=0.1 max_grad_E_norm=1000 \
#     annealing_beta0=1.0 annealing_beta1=0.001 annealing_type=langevin \
#     save_freq=1000 optim.beta1=0.5

# srun pwd && SLURM_GPUS_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 python eval.py experiment=lj55 \
#     resample_batch_size=10000 num_epochs_per_stage=500 \
#     beta1=0.1 \
#     max_grad_E_norm=1000 annealing_beta0=1.0 annealing_beta1=0.001 annealing_type='langevin' \
#     save_freq=500 seed=0
#     checkpoint=../../2025-09-20/19-54-23/checkpoints/checkpoint_latest.pt