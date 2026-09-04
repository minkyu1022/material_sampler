import logging
import torch.distributed as dist
import matplotlib
import torch
import numpy as np
import scipy as sp
import wandb
from omegaconf import OmegaConf, DictConfig
import os


@torch.no_grad()
def sample_categorical_logits(logits, dtype=torch.float64):
    # do not require logits to be log-softmaxed
    gumbel_noise = -(1e-10 - (torch.rand_like(logits, dtype=dtype) + 1e-10).log()).log()
    return (logits + gumbel_noise).argmax(dim=-1)


def ess(log_rnd, normalize=True): 
    """
    log_rnd: [B]
    Compute effective sample size:
        If normalize: divide ESS by batch size, so range is [0, 1]; 
        otherwise, range is [0, B]
    """
    weights = log_rnd.detach().softmax(dim=-1)
    ess = 1 / (weights ** 2).sum().item()
    return ess / log_rnd.shape[0] if normalize else ess


def get_local_logger(logpath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    if saving: # log to file
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        info_file_handler.setFormatter(formatter)
        logger.addHandler(info_file_handler)
    if displaying: # log to console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


class Logger:
    def __init__(self, cfg: DictConfig, rank: int = 0, local_log_on_all: bool = False):
        # if local_log_on_all, all ranks will log locally; otherwise, only rank 0 will log locally
        self.rank = rank # for distributed training

        if self.rank == 0 or local_log_on_all:
            self.local_logger = get_local_logger(logpath=os.path.join(cfg.logging.dir, '.log'), 
                                                 displaying=cfg.logging.displaying,
                                                 saving=cfg.logging.saving,
                                                 debug=cfg.logging.debug)
        else:
            self.local_logger = None

        if self.rank == 0 and cfg.logging.use_wandb:
            assert wandb.login(key=cfg.logging.wandb_api_key)
            self.wandb_logger = wandb.init(
                entity=cfg.logging.wandb_entity,
                project=cfg.logging.wandb_project,
                name=str(cfg.logging.run_name),
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                settings=wandb.Settings(init_timeout=300),
                dir=os.environ.get("WANDB_DIR", None) # it is recommended to set WANDB_DIR env variable
            )
        else:
            self.wandb_logger = None

    def info(self, msg: str):
        """Log an info message only to the local logger."""
        if self.local_logger is not None:
            self.local_logger.info(msg)

    def log(self, data: dict, step=None):
        """Log key-value pairs to local logger and wandb logger (if enabled)."""
        if self.local_logger is not None or self.wandb_logger is not None:
            msg = []
            for k, v in sorted(data.items()):
                if isinstance(v, (int, float, str)):
                    msg.append(f'{k}: {v:.4f}' if isinstance(v, float) else f'{k}: {v}')
                elif isinstance(v, matplotlib.figure.Figure):
                    data[k] = wandb.Image(v)
            if msg and self.local_logger is not None:
                self.local_logger.info(f"Step {step}: " + ', '.join(msg))
            if self.wandb_logger is not None:
                self.wandb_logger.log(data, step=step)

    def close(self, exit_code=0):
        """Close the wandb logger."""
        if self.wandb_logger is not None:
            self.wandb_logger.finish(exit_code=exit_code)
        if dist.is_initialized():
            dist.destroy_process_group()


def get_mu_from_coeff(coeffs):
    r"""
    Shape [K] -> [K]
    mu(k) = \prod_{i=1}^{k} (1 - coeff(i))
    The boundary values coeff(0) = 0 and mu(0) = 1 are not included in both input and output, but can be added manually.
    """
    mus = [1]
    for i in range(len(coeffs)):
        mus.append(mus[-1] * (1 - coeffs[i]))
    return mus[1:]


def get_coeff_from_mu(mus):
    r"""
    Shape [K] -> [K]
    coeff(k) = 1 - mu(k) / mu(k-1)
    The boundary values coeff(0) = 0 and mu(0) = 1 are not included in both input and output
    """
    coeffs = [1 - mus[0]]
    for i in range(1,len(mus)):
        coeffs.append(1 if mus[i-1] == 0.0 else 1 - mus[i] / mus[i-1])
    return coeffs


def get_eta_from_coeff(coeffs, alpha=1):
    r"""
    Shape [K] -> [K]
    eta(k) = coeff(k) / (alpha * (1 - coeff(k))
    coeff=0 means eta=0, coeff=1 means eta=inf
    """
    if not isinstance(coeffs, np.ndarray):
        coeffs = np.array(coeffs)
    return np.where(np.array(coeffs) == 1, np.inf, coeffs / (alpha * (1 - coeffs) + 1e-10))


def find_coeff_kl(log_rnd_running, reward, mu, target_kl, pdns_loss_version='v1', logger: Logger=None, default_coeff: float = 0.1):
    r"""
    Find coeff such that the KL of the new weights is close to target_kl.

    Args:
        log_rnd_running: log_rnd(x) = \log\frac{dP^0}{dP^\theta_{k-1}} (x), [bs]
        reward: r(x), [bs]
        mu: mu(k-1), a float
        target_kl: target kl, a float > 0

    Return:
        coeff: a float
    """
    log_rnd_running = log_rnd_running.float().cpu().numpy(); reward = reward.float().cpu().numpy()
    bsz = log_rnd_running.shape[0]

    def loss(coeff):
        if pdns_loss_version == 'v1':
            weights = sp.special.softmax(log_rnd_running + (1 - mu * (1 - coeff)) * reward, axis=0)
        else:
            weights = sp.special.softmax(coeff * (log_rnd_running + reward), axis=0)
        kl = -np.mean(np.log(bsz * weights))
        return (kl - target_kl) ** 2

    result = sp.optimize.minimize(loss, x0=0.5, bounds=[[0.01, 0.99]])

    if not result.success:
        # with np.printoptions(threshold=float('inf')):
        logger.info("Optimizer in `find_coeff` returned FAIL!\n"
                    f">> Optimizer message:\n{result.message}\n"
                    f">> log_rnd_running:\n{log_rnd_running}\n"
                    f">> reward:\n{reward}\n"
                    f"Use default coeff {default_coeff} instead.")
        return default_coeff
    else:
        if result.x.item() < 0.02 or result.x.item() > 0.98:
            # with np.printoptions(threshold=float('inf')):
            logger.info(f"Optimizer in `find_coeff` returned coeff {result.x.item():.4f} near the boundary of [0, 1], "
                        "which may require further investigation.\n"
                        f">> Optimizer message:\n{result.message}\n"
                        f">> log_rnd_running:\n{log_rnd_running}\n"
                        f">> reward:\n{reward}\n")
        return result.x.item()

def gather_across_processes(tensor, world_size=1, device='cuda:0'):
    """
    Gather a tensor across all processes in the distributed training.
    tensor: Tensor of shape [B, ...] where B is the batch size.
    Returns a tensor of shape [B * world_size, ...] where world_size is the number of processes.
    """
    if world_size == 1:
        return tensor
    gathered_tensor = torch.empty([tensor.shape[0] * world_size, *tensor.shape[1:]], 
                                  dtype=tensor.dtype, device=device)
    dist.all_gather_into_tensor(gathered_tensor, tensor)
    return gathered_tensor
