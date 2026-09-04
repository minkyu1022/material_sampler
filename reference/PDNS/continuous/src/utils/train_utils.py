from mailbox import NotEmptyError
import os
import json
from typing import Any
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import PIL
import wandb

import torch
from torch.optim import Optimizer

import src.utils.distributed_mode as distributed_mode
from src.components.matchers import ScoreMatcher as Matcher
from src.components.buffer import BatchBuffer

def setup(cfg):
    print("Found {} CUDA devices.".format(torch.cuda.device_count()))
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(
            "{} \t Memory: {:.2f}GB".format(
                props.name, props.total_memory / (1024**3)
            )
        )
    print(dict(os.environ))
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))

    distributed_mode.init_distributed_mode(cfg)

    if distributed_mode.is_main_process():
        args_filepath = Path("cfg.yaml")
        print(f"Saving cfg to {args_filepath}")
        with open("config.yaml", "w") as fout:
            print(OmegaConf.to_yaml(cfg), file=fout)
        with open("env.json", "w") as fout:
            print(json.dumps(dict(os.environ)), file=fout)


def is_init_stage(epoch: int, cfg: DictConfig):
    n_a = cfg.adjoint_matcher.num_epochs_per_stage
    n_b = cfg.bridge_matcher.num_epochs_per_stage

    assert cfg.init_stage in ("adjoint", "bridge")
    n = n_a if cfg.init_stage == "adjoint" else n_b
    return epoch < n


def is_init_epoch_per_stage(epoch: int, cfg: DictConfig):
    n_a = cfg.adjoint_matcher.num_epochs_per_stage
    n_b = cfg.bridge_matcher.num_epochs_per_stage

    assert cfg.init_stage in ("adjoint", "bridge")
    if cfg.init_stage == "adjoint":
        return epoch == 0 or epoch == n_a
    else:
        return epoch == 0 or epoch == n_b


def determine_stage(epoch: int, cfg: DictConfig):
    n_a = cfg.adjoint_matcher.num_epochs_per_stage
    n_b = cfg.bridge_matcher.num_epochs_per_stage

    if cfg.init_stage == "adjoint":
        return "adjoint" if (epoch % (n_a + n_b)) < n_a else "bridge"

    elif cfg.init_stage == "bridge":
        return "bridge" if (epoch % (n_a + n_b)) < n_b else "adjoint"
    else:
        raise NotImplementedError


def determine_stage_NAS(epoch: int, cfg: DictConfig):
    n_a = cfg.adjoint_matcher1.num_epochs_per_stage
    n_b = cfg.adjoint_matcher2.num_epochs_per_stage

    if cfg.init_stage == "adjoint1":
        return "adjoint1" if (epoch % (n_a + n_b)) < n_a else "adjoint2"

    elif cfg.init_stage == "adjoint2":
        return "adjoint2" if (epoch % (n_a + n_b)) < n_b else "adjoint1"
    else:
        raise NotImplementedError

def n_resample_batches(
    matcher: Matcher,
    batch_size: int,
    epoch: int,
    cfg: DictConfig,
) -> int:
    if cfg.chem:
        # fix the resampled samples per local gpu
        is_init_epoch = is_init_epoch_per_stage(epoch, cfg)
        N0, N = matcher.init_resample_size, matcher.resample_size
        return (N0 if is_init_epoch else N) // batch_size
    else:
        # fix the total resampled samples across all gpus
        return matcher.resample_size // (batch_size * cfg.world_size)


def dryrun_overwrite(cfg):
    assert cfg.dryrun

    cfg.project = f"{cfg.project}_dryrun"

    cfg.nfe = 10
    cfg.eval_freq = 1

    cfg.adjoint_matcher.duplicates = 1
    cfg.adjoint_matcher.num_epochs_per_stage = 1

    cfg.bridge_matcher.duplicates = 1
    cfg.bridge_matcher.num_epochs_per_stage = 1

    if cfg.chem:
        cfg.adjoint_matcher.init_resample_size = cfg.adjoint_matcher.resample_size
        cfg.bridge_matcher.init_resample_size = cfg.bridge_matcher.resample_size


def save(
    epoch: int,
    controller: torch.nn.Module,
    buffer: BatchBuffer,
    optimizer: Optimizer,
    cfg: DictConfig,
    ckpt_dir: Path = Path("checkpoints"),
):
    ckpt_dir.mkdir(exist_ok=True)

    state = {
        "epoch": epoch,
        "cfg": cfg,
        "optimizer": optimizer.state_dict(),
        "buffer": buffer.state_dict() if buffer is not None else None,
    }

    if cfg.distributed:
        try:
            state["controller"] = controller.module.state_dict()
        except:
            state["controller"] = controller.state_dict()
    else:
        state["controller"] = controller.state_dict()

    # torch.save(state, ckpt_dir / "checkpoint_{}.pt".format(epoch)) ## I remove this just for to save our memory space
    torch.save(state, ckpt_dir / "checkpoint_latest.pt")


def load(
    checkpoint: dict[str: Any],
    controller: torch.nn.Module,
    buffer: BatchBuffer,
    optimizer: Optimizer,
):
    controller.load_state_dict(checkpoint["controller"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    buffer.load_state_dict(checkpoint["buffer"])
    return checkpoint["epoch"] + 1


class Writer:
    def __init__(self, name: str, cfg: DictConfig, is_main_process: bool):
        if cfg.use_wandb and is_main_process:
            # wandb.login(host='https://fairwandb.org/')
            self.writer = wandb.init(
                mode='online',
                project=cfg.project,
                name=name,
                config=dict(cfg),
            )
        else:
            self.writer = None

    def log(self, log_dict: dict, step=None):
        if self.writer is not None:
            # convert images
            for k in log_dict.keys():
                if isinstance(log_dict[k], PIL.Image.Image):
                    log_dict[k] = wandb.Image(log_dict[k])
            self.writer.log(log_dict, step=step)


def compute_gradient_norm(model, norm_type=2):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm