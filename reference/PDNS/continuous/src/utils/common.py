from __future__ import annotations

import os
import math
import requests
import hydra
from pathlib import Path

import torch
# import wandb


def get_timesteps(
    t0: torch.Tensor | float,
    t1: torch.Tensor | float,
    dt: torch.Tensor | float | None = None,
    steps: int | None = None,
    rescale_t: str | None = None
) -> torch.Tensor:
    if (steps is None) is (dt is None):
        raise ValueError("Exactly one of `dt` and `steps` should be defined.")
    if steps is None:
        steps = int(math.ceil((t1 - t0) / dt))
    if rescale_t is None:
        return torch.linspace(t0, t1, steps=steps)
    elif rescale_t == "quad":
        return torch.sqrt(
            torch.linspace(t0, t1.square(), steps=steps)
        ).clip(max=t1)
    elif rescale_t == "cosine":
        """
        Copied verbatim from
        https://github.com/franciscovargas/denoising_diffusion_samplers/blob/main/dds/discretisation_schemes.py#L50
        """
        s = 0.008  # Choice from original paper
        pre_phase = torch.linspace(t0, t1, steps) / t1
        phase = ((pre_phase + s) / (1 + s)) * torch.pi * 0.5

        dts = torch.cos(phase) ** 4

        dts /= dts.sum()
        dts *= t1  # We normalise s.t. \sum_k \beta_k = T (where beta_k = b_m*cos^4)

        dts_out = torch.concat(
            (torch.tensor([t0]), torch.cumsum(dts, -1))
        )

        return dts_out
    raise ValueError("Unkown timestep rescaling method.")


def download_github_file(repo_owner, repo_name, file_path):
    root = Path(hydra.utils.get_original_cwd())
    save_path = root / "data" / Path(file_path).name

    if save_path.exists():
        print(f"File {file_path} aleady downloaded")
    else:
        url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/main/{file_path}"
        response = requests.get(url)
        if response.status_code == 200:
            save_path.parent.mkdir(exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(response.content)
            print(f"File {file_path} downloaded successfully")
        else:
            print(f"Failed to download file {file_path}. Status code: {response.status_code}")

    return save_path
