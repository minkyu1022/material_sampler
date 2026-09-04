from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics.aggregation import MeanMetric

from src.components.matchers import ScoreMatcher
from src.components.sdes import BaseSDE
from src.energies.dist_energy import DistEnergy

def train_one_epoch(
    dataloader: DataLoader,
    matcher: ScoreMatcher,
    model: torch.nn.Module,
    sde: BaseSDE,
    source: DistEnergy,
    optimizer: Optimizer,
    lr_schedule: LRScheduler | None,
    device: str,
    cfg: DictConfig,
):

    epoch_loss = MeanMetric().to(device, non_blocking=True)
    model_grad_norm = MeanMetric().to(device, non_blocking=True)

    # training loop
    model.train(True)
    for idx, data in enumerate(dataloader):
        optimizer.zero_grad()
        input, sigma, sigma_total, score, weight = matcher.prepare_target(data, sde, device)
        weight = torch.clamp(weight, min=cfg.clip_adv_min, max=cfg.clip_adv_max)
        output = model(*input)

        if cfg.param_type == 'control' and cfg.loss_type == 'control':
            target = sigma * score
            loss = (weight * ((output - target)**2).mean(1)).mean()
        
        elif cfg.param_type == 'adjoint' and cfg.loss_type == 'control':
            loss = (weight * ((sigma*(output - score))**2).mean(1)).mean()
        
        elif cfg.param_type == 'adjoint' and cfg.loss_type == 'adjoint':
            loss = (weight * ((output - score)**2).mean(1)).mean()
        
        elif cfg.param_type == 'adjoint' and cfg.loss_type == 'epsilon':
            loss = (weight * (sigma_total*(output - score)**2).mean(1)).mean()
        
        elif cfg.param_type == 'control' and cfg.loss_type == 'epsilon':
            target = sigma * score
            loss = (weight * (sigma_total*(output - target)**2).mean(1)).mean()
        
        else:
            raise NotImplementedError

        loss.backward()

        if cfg.clip_grad_norm:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.clip_grad_norm)

        optimizer.step()

        if cfg.ema_decay > 0:
            ema_model = model.module if cfg.distributed else model
            ema_model.update_ema()

        epoch_loss.update(loss.item())
        model_grad_norm.update(norm)
        if lr_schedule:
            lr_schedule.step()

    
    return float(epoch_loss.compute().detach().cpu()), float(model_grad_norm.compute().detach().cpu())