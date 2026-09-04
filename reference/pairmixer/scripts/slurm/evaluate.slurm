#!/bin/bash
#SBATCH --job-name evaluate

### Logging
#SBATCH --output logs/evaluation/slurm_%j.out
#SBATCH --error  logs/evaluation/slurm_%j.err

### Node info (update these for your cluster)
#SBATCH --account YOUR_ACCOUNT
#SBATCH --nodes 32
#SBATCH --partition YOUR_PARTITION
#SBATCH --ntasks-per-node=1
#SBATCH --time 48:00:00

hostname
source ~/.bashrc
module load gcc
conda init
conda activate pairmixer
cd ~/code/pairmixer

DATASET=${1:-casp15}
NAME=${2:-boltz}

echo "Running with parameters:"
echo "DATASET: $DATASET"
echo "NAME: $NAME"

evaluate_start_time=$(date +%s)

if [ "$DATASET" == "all" ]; then
  for ds in casp15 test; do
    cmd="srun python scripts/eval/run_evals.py \
      logs/boltz_results_final/predictions/${ds}/${NAME}/boltz_results_queries/predictions \
      logs/boltz_results_final/targets/${ds} \
      logs/boltz_results_final/evals/${ds}/${NAME} \
      --testset ${ds}
    "
    echo $cmd
    eval $cmd
  done
else
  cmd="srun python scripts/eval/run_evals.py \
    logs/boltz_results_final/predictions/${DATASET}/${NAME}/boltz_results_queries/predictions \
    logs/boltz_results_final/targets/${DATASET} \
    logs/boltz_results_final/evals/${DATASET}/${NAME} \
    --testset ${DATASET}
  "
  echo $cmd
  eval $cmd
fi

evaluate_end_time=$(date +%s)
evaluate_duration=$((evaluate_end_time - evaluate_start_time))
echo "Evaluate short duration: $evaluate_duration seconds"
