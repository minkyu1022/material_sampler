# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import sys
import math
import traceback
import hydra
import numpy as np
import termcolor
from tqdm import tqdm
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

# main components
from src.utils.common import get_timesteps
from src.components.sdes import VESDE, VPSDE, AnnealedSDE, ControlledSDE
from src.components.matchers import ScoreMatcher
from src.components.term_cost import TermCost
from src.train_loop import train_one_epoch
from src.eval_loop import eval_epoch

# utils
import src.utils.train_utils as train_utils

from src.utils.ema import EMA
import src.utils.distributed_mode as distributed_mode
from ipdb import set_trace as debug
cudnn.benchmark = True


def red(content): return termcolor.colored(str(content),"red",attrs=["bold"])
def green(content): return termcolor.colored(str(content),"green",attrs=["bold"])
def blue(content): return termcolor.colored(str(content),"blue",attrs=["bold"])
def cyan(content): return termcolor.colored(str(content),"cyan",attrs=["bold"])
def yellow(content): return termcolor.colored(str(content),"yellow",attrs=["bold"])
def magenta(content): return termcolor.colored(str(content),"magenta",attrs=["bold"])


@hydra.main(config_path="configs", config_name="train.yaml", version_base="1.1")
def main(cfg):
    try:
        train_utils.setup(cfg)
        if cfg.dryrun:
            train_utils.dryrun_overwrite(cfg)
        print(str(cfg))

        device = "cuda"

        # fix the seed for reproducibility
        seed = cfg.seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        print("Instantiating energy...")
        energy = hydra.utils.instantiate(cfg.energy, device=device)


        print("Instantiating source...")
        if cfg.chem:
            source = hydra.utils.instantiate(cfg.source, device=device)
            eval_source = hydra.utils.instantiate(
                cfg.eval_source,
                energy=energy,
                device=device
            )
        else:
            source = hydra.utils.instantiate(cfg.source, device=device)
            eval_source = source

        print('Instantiating model...')
        ref_sde = hydra.utils.instantiate(cfg.ref_sde).to(device)
        controller = hydra.utils.instantiate(cfg.controller, score=energy.score, prior_score=source.score).to(device)
        if cfg.ema_decay > 0:
            controller = EMA(controller, cfg.ema_decay)

        print("Instantiating eta and matchers...")
        gamma = hydra.utils.instantiate(cfg.gamma_scheduler, device=device)
        if isinstance(ref_sde, VESDE): sigma = math.sqrt(cfg.sigma_max**2 - cfg.sigma_min**2)
        else: sigma = cfg.sigma
        term_cost_fn = TermCost(energy=energy, sigma=sigma, clip_term_norm=cfg.clip_term_norm)

        matcher = ScoreMatcher(source=source, **cfg.score_matcher, 
                               term_cost=term_cost_fn, gamma=gamma, alpha=cfg.alpha, 
                               iws=cfg.iws, cumulate=cfg.cumulate)

        print("Instantiating optimizer...")
        lr_schedule = None # TODO: Scheduler
        optimizer = torch.optim.Adam(controller.parameters(), 
                                     lr=cfg.optim.lr, 
                                     betas=(cfg.optim.beta1, cfg.optim.beta2),
                                     weight_decay=cfg.optim.weight_decay)

        # Load checkpoint
        checkpoint_path = Path(cfg.checkpoint or "checkpoints/checkpoint_latest.pt")
        checkpoint_path.parent.mkdir(exist_ok=True)
        if checkpoint_path.exists():
            print(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path)
            start_epoch = train_utils.load(checkpoint, controller, matcher.buffer, optimizer,)
            stage = 0
        else:
            start_epoch = 0
            stage = -1

        if cfg.distributed:
            controller = torch.nn.parallel.DistributedDataParallel(
                controller, device_ids=[cfg.gpu], find_unused_parameters=True
            )

        print("Instantiating SDE...")
        sde = ControlledSDE(ref_sde, controller, cfg.param_type).to(device)


        print("Instantiating writer...")
        writer = train_utils.Writer(
            name=cfg.exp or f'{source.name}_to_{energy.name}',
            cfg=cfg,
            is_main_process=distributed_mode.is_main_process(),
        )        


        ## Training loop starts
        print(f"Starting from {start_epoch}/{cfg.num_epochs} epochs...")        
        for epoch in range(start_epoch, cfg.num_epochs):
            # 1. generate buffer, make dataset
            if epoch % cfg.num_epochs_per_stage == 0 or epoch == start_epoch:
                stage += 1

                # 1.1 eval mode             
                controller.train(False)

                # 1.2 clean buffer
                matcher.clean_buffer()

                # 1.3 Resample
                print('Resampling...')
                
                B, N = cfg.resample_batch_size, cfg.score_matcher.buffer_size
                assert  N % B == 0
                M = N // B

                ### if annealed sample with annealed_sde
                if epoch == 0 and cfg.annealed:
                    print('Use annealed SDE')
                    annealed_sde = hydra.utils.instantiate(cfg.annealed_sde, energy=energy, source=source).to(device)
                
                for _ in range(M):
                    x0 = source.sample([B,]).to(device)
                    timesteps = get_timesteps(**cfg.timesteps).to(device)

                    # generate sample with old control
                    if epoch == 0 and cfg.annealed: ## annealed dynamics
                        matcher.populate_buffer(x0, annealed_sde, timesteps, cfg.zero_last_step_noise)
                    else:
                        matcher.populate_buffer(x0, sde, timesteps, cfg.zero_last_step_noise) 

                print('Resampling done!')
                
                if cfg.annealed and epoch == 0:
                    print('Start evaluation')
                    eval_dict = eval_epoch(
                                f"annealed",
                                annealed_sde,
                                eval_source,
                                energy,
                                term_cost_fn,
                                gamma,
                                matcher.beta,
                                cfg,
                                device=device
                            )
                    
                    eval_dir = Path("eval_figs")
                    eval_dir.mkdir(exist_ok=True)
                    if "marginal" in eval_dict:
                        eval_dict["marginal"].save(eval_dir / f"marginal_{epoch}.png")
                    if "energy_hist" in eval_dict:
                        eval_dict["energy_hist"].save(eval_dir / f"energy_hist_{epoch}.png")

                    writer.log(eval_dict, step=epoch)
                    print('Evaluation done!')

                # 1.4 make it into dataloader
                dataloader, buffer_data, gamma_value = matcher.build_dataloader(cfg.train_batch_size, stage, adapt_scheduler=True)

                # 1.5 logging                
                x, running_cost, term_cost, weight = buffer_data['x1'], buffer_data['log_rnd'], buffer_data['term_cost'], buffer_data['weight']
                writer.log({'running_cost': running_cost.mean(),
                            'term_cost_mean': term_cost.mean(),
                            'term_cost_max': term_cost.max(),
                            'term_cost_min': term_cost.min(),
                            'weight_mean': weight.mean(),
                            'weight_max': weight.max(),
                            'weight_min': weight.min(),
                            'gamma': gamma_value
                            }, step=epoch)
                
                # 1.6 initialize (overwrite) controller/optimizer 
                controller.train(True)
                if cfg.ema_decay and start_epoch < epoch:
                    for param, param_ema  in zip(controller.model.parameters(), controller.shadow_params.parameters()):
                        param.data.copy_(param_ema.data)

                optimizer = torch.optim.Adam(controller.parameters(), 
                                     lr=cfg.optim.lr, 
                                     betas=(cfg.optim.beta1, cfg.optim.beta2),
                                     weight_decay=cfg.optim.weight_decay)
                
                print('Training loop...')
            
            # 1.7 If we use importance weighted sampling (IWS), get new IWS samples
            elif cfg.iws:
                dataloader, buffer_data, gamma_value = matcher.build_dataloader(cfg.train_batch_size, stage, adapt_scheduler=False)
                writer.log({'gamma': gamma_value}, step=epoch)
            

            # 2. Learn control
            loss, grad_norm = train_one_epoch(
                dataloader,
                matcher, # Buffer that has (X_1, weight), and prepare target
                controller, # current trainable model
                sde, # use sde only for posterior sampling, i.e. X_t ~ p_t ( | X_0, X_1)
                source,
                optimizer,
                lr_schedule,
                device,
                cfg
            )

            # train dict
            train_dict = {f"loss": loss, 
                          f"buffer_size": len(matcher.buffer), 
                          f"grad_norm": grad_norm}
            writer.log(train_dict, step=epoch)

            # save checkpoint
            if distributed_mode.is_main_process() and epoch > 0 and (
                epoch % cfg.save_freq == 0 or epoch + 1 == cfg.num_epochs
            ):
                print("Saving checkpoint ... ")
                train_utils.save(
                    epoch,
                    controller,
                    matcher.buffer,
                    optimizer,
                    cfg,
                )

            # evaluation
            if distributed_mode.is_main_process() and epoch > 0 and (
                epoch % cfg.eval_freq == 0 or epoch + 1 == cfg.num_epochs
            ):
                print("Evaluation ... ")
                try:
                    with torch.no_grad():
                        controller.train(False)
                        eval_dict = eval_epoch(
                            f"ep{epoch}",
                            sde,
                            eval_source,
                            energy,
                            term_cost_fn,
                            gamma,
                            matcher.beta,
                            cfg,
                            device=device
                        )
                    
                    eval_dir = Path("eval_figs")
                    eval_dir.mkdir(exist_ok=True)
                    if "marginal" in eval_dict:
                        eval_dict["marginal"].save(eval_dir / f"marginal_{epoch}.png")
                    if "energy_hist" in eval_dict:
                        eval_dict["energy_hist"].save(eval_dir / f"energy_hist_{epoch}.png")

                    writer.log(eval_dict, step=epoch)
                    controller.train(True)

                except Exception as e:
                    # Log exception but don't stop training.
                    print(traceback.format_exc())
                    print(traceback.format_exc(), file=sys.stderr)
                print("Evaluation done!")

    except Exception as e:
        # This way we have the full traceback in the log.  otherwise Hydra
        # will handle the exception and store only the error in a pkl file
        print(traceback.format_exc(), file=sys.stderr)
        raise e


if __name__ == "__main__":
    main()
