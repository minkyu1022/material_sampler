# Discrete Sampling Experiments

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2510.03824)
[![Paper](https://img.shields.io/badge/Paper-ICLR2026-blue)](https://openreview.net/forum?id=XTHQqS7ObC)

This folder contains the official implementation of the Proximal Diffusion Neural Sampler (PDNS) for discrete sampling experiments.

## Installation

We require the following dependencies to run the code:

```{bash}
pip install numpy, matplotlib, torch, tqdm, einops, omegaconf, timm, pyyaml, wandb
```

## Usage

```bash
python train.py \
    --config ${CONFIG} \
    logging.run_name=${RUN_NAME} \
    ${PARAMS} \
```

- `CONFIG`: the name of the config file in `configs/`, e.g., `ising4.yaml`.
- `RUN_NAME`: the name of the run, e.g., `ising4`. All logs and checkpoints will be saved in `logging.dir` (see configs for details).
- `PARAMS`: other parameters to override the config file.

If one needs to resume training from a checkpoint, add the following argument: `ckpt_path=${CKPT_PATH}`, where `CKPT_PATH` is the path to the checkpoint (`.pth` file), e.g., `outputs/ising4/ckpt1000.pth`. The training will resume from the checkpoint and all logs will be saved in the same directory as the checkpoint.

Further details on the arguments can be found in `configs/`. Please also ensure to replace `your_wandb_entity` and `your_wandb_project` in the config files with your actual wandb entity and project name to enable logging.
