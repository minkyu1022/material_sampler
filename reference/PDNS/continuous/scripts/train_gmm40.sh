#!/bin/bash
#SBATCH --job-name=gmm40
#SBATCH --gres=gpu:H100:1
#SBATCH --partition=coe-gpu
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=32
#SBATCH --output=./sbatch_logs_recent/O-%x.%j
#SBATCH --error=./sbatch_logs_recent/E-%x.%j

source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate /home/hice1/jchoi843/scratch/anaconda3/envs/pce-dm

srun pwd && SLURM_GPUS_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 python train.py experiment=gmm40 exp=gmm40_run2
