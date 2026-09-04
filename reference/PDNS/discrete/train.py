import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import timedelta
import os
from model import get_rope_vit_model, EMA
from utils_ising import ising2d_ham
from utils_potts import potts2d_ham
from utils import Logger
from utils_train import train_pdns, train_pdns_adaptive
from omegaconf import OmegaConf
if not OmegaConf.has_resolver('eval'): OmegaConf.register_new_resolver('eval', eval)
import traceback
import argparse


def main(args):
    args.world_size = world_size = torch.cuda.device_count()
    # Reminder: if not on slurm system, always set CUDA_VISIBLE_DEVICES before run!
    if world_size > 1:
        port = int(np.random.randint(10000, 20000))
        mp.set_start_method("forkserver")
        mp.spawn(run_multiprocess, args=(world_size, args, port), nprocs=world_size, join=True)
    else:
        run_multiprocess(0, 1, args, None)

def run_multiprocess(rank, world_size, args, port):
    if world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(minutes=30))
        torch.cuda.set_device(rank)
    args.device = f'cuda:{rank}'
    logger = Logger(args, rank=rank)
    logger.info(f"Using {world_size} GPU(s) for training.")

    if args.dist == 'ising':
        def reward_fn(S, beta=args.beta, J=args.J, h=args.h):
            return -beta * ising2d_ham(2*S-1, J, h)
    elif args.dist == 'potts':
        def reward_fn(S, beta=args.beta, J=args.J):
            return -beta * potts2d_ham(S, J)
    else:
        raise ValueError(f"Unknown target distribution: {args.dist}")

    assert args.model.name == 'ropevit', f"Unknown model name: {args.model.name}"
    model = get_rope_vit_model(args).to(device=args.device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank], static_graph=True)
    ema = EMA(model.parameters(), decay=args.ema.decay)

    model_info = 'Model: num of params: {}, size: {:.2f} MB'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2))
    logger.info(model_info)

    args.start_step_stage = 1
    if args.ckpt_path is not None:
        ckpt = torch.load(args.ckpt_path, weights_only=False, map_location=args.device)
        model_module = model.module if hasattr(model, 'module') else model # for DDP
        model_module.load_state_dict(ckpt['model_state_dict'])
        ema.load_state_dict(ckpt['ema_state_dict'])
        logger.info(f'Checkpoint loaded from {args.ckpt_path}')

        # ckpt_path should be like xxx/ckpt?.pth
        args.start_step_stage = int(args.ckpt_path.split('ckpt')[-1].split('.pth')[0]) + 1
        logger.info(f"Resuming from stage {args.start_step_stage}")

    trainer = train_pdns_adaptive if args.pdns_adaptive else train_pdns

    # check the validity of some cfg options
    if args.wdce_antithetic:
        assert args.wdce_num_replicates % 2 == 0, "wdce_num_replicates must be even when using antithetic sampling"
    assert args.wdce_loss_formulation in ['lambda', 'm'], f"Unknown wdce_loss_formulation: {args.wdce_loss_formulation}"
    assert args.pdns_loss_version in ['v1', 'v2'], f"Unknown pdns_loss_version: {args.pdns_loss_version}"

    if world_size > 1:
        logger.info(
            f"All batch and buffer sizes are *per device* in DDP mode.\n"
            f"This means the effective batch size is {args.batch_size * world_size=},\n"
            f"the effective evaluation batch size is {args.eval_batch_size * world_size=},\n"
            f"and the effective buffer size is {args.buffer_size * world_size=}"
        )

    try:
        trainer(model, reward_fn, ema, args, logger)
        logger.close()
    except Exception as e:
        error_info = traceback.format_exc()
        logger.info(f">>> Training failed with error:\n{error_info}")
        logger.close(exit_code=1)
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file")
    known_args, unknown_args = parser.parse_known_args()
    config = OmegaConf.load('configs/' + known_args.config)
    overrides = OmegaConf.from_dotlist(unknown_args)
    args = OmegaConf.merge(config, overrides)
    OmegaConf.set_struct(args, False) # Allow dynamic updates to args
    os.makedirs(args.logging.root_dir, exist_ok=True)
    os.makedirs(args.logging.dir, exist_ok=True)
    main(args)