#!/bin/bash
#SBATCH --job-name train

### Logging
#SBATCH --output logs/slurm/train_%j.out
#SBATCH --error  logs/slurm/train_%j.err

### Node info (update these for your cluster)
#SBATCH --account YOUR_ACCOUNT
#SBATCH --nodes 64
#SBATCH --partition YOUR_PARTITION
#SBATCH --ntasks-per-node=1
#SBATCH --time 48:00:00

nvidia-smi
hostname

source ~/.bashrc
module load gcc cuda/12.4

conda init
conda activate pairmixer
cd ~/code/pairmixer

### Training
train_start_time=$(date +%s)
srun python scripts/train/train.py $1
train_end_time=$(date +%s)
train_time=$((train_end_time - train_start_time))
echo "Training took $((train_time/3600)) hours, $(((train_time%3600)/60)) minutes"

