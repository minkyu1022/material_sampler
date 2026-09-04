#!/bin/bash
#SBATCH --job-name predict

### Logging
#SBATCH --output logs/prediction/slurm_%j.out
#SBATCH --error  logs/prediction/slurm_%j.err

### Node info (update these for your cluster)
#SBATCH --account YOUR_ACCOUNT
#SBATCH --nodes 32
#SBATCH --partition YOUR_PARTITION
#SBATCH --ntasks-per-node=1
#SBATCH --time 04:00:00

nvidia-smi
hostname
source ~/.bashrc
module load gcc
conda init
conda activate pairmixer
cd ~/code/pairmixer

DATASET=${1:-casp15}
NAME=${2:-boltz}
shift 2  # Remove first two arguments
ADD_ARGS="$@"  # All remaining arguments

echo "Running with parameters:"
echo "DATASET: $DATASET"
echo "NAME: $NAME"
echo "ADDITIONAL_ARGS: $ADD_ARGS"

predict_start_time=$(date +%s)

if [ "$DATASET" == "all" ]; then
  for DATASET in test casp15; do
    cmd="srun pairmixer predict \
                logs/boltz_results_final/inputs/${DATASET}/boltz/queries \
      --out_dir logs/boltz_results_final/predictions/${DATASET}/${NAME} \
      --recycling_steps 10 \
      --sampling_steps 200 \
      --diffusion_samples 5 \
      --seed 42 \
      --preprocessing-threads 1 \
      $ADD_ARGS
    "
    echo $cmd
    eval $cmd
  done
else
  cmd="srun pairmixer predict \
              logs/boltz_results_final/inputs/${DATASET}/boltz/queries \
    --out_dir logs/boltz_results_final/predictions/${DATASET}/${NAME} \
    --recycling_steps 10 \
    --sampling_steps 200 \
    --diffusion_samples 5 \
    --seed 42 \
    --preprocessing-threads 1 \
    $ADD_ARGS
  "
  echo $cmd
  eval $cmd
fi

predict_end_time=$(date +%s)
predict_duration=$((predict_end_time - predict_start_time))
echo “Predict duration: $predict_duration seconds”
