from functools import partial
import torch
import torch.distributed as dist
from model.ema import EMA
from utils import ess, sample_categorical_logits, find_coeff_kl, Logger, get_mu_from_coeff, gather_across_processes
from utils_ising import ising2d_mag, ising2d_plot_2pt_corr, ising2d_visualize
from utils_potts import potts2d_mag, potts2d_plot_2pt_corr, potts2d_visualize
import numpy as np
from tqdm import tqdm
import os


def rnd_mask(model, reward_model, batch_size, device='cuda:0'):
    r"""
    Run random order sampling and compute the RND $\log\frac{dP^*}{dP^u}$ along the trajectory
    model: the model s_{\theta_u} that is being fine-tuned
    reward_model: r(X)

    \log\frac{dP^*}{dP^u}(X) = \log\frac{dP^0}{dP^u}(X) + \log\frac{dP^*}{dP^0}
                             = \log\frac{dP^0}{dP^u}(X) + reward_model(X_T) - log Z
             ==> W^u - log Z = log_rnd_running + rwd - log Z

    return:
    - x: the final samples, [B, D]
    - log_rnd_running: the log RND along the trajectory, [B]
    - rwd: the reward for the final samples, [B]
    """
    model_module = model.module if hasattr(model, 'module') else model # for DDP
    x = torch.full((batch_size, model_module.length), model_module.vocab_size-1, device=device, dtype=torch.int64)
    batch_arange = torch.arange(batch_size, device=device)
    jump_pos = torch.rand(x.shape, device=device).argsort(dim=-1)
    log_rnd_running = torch.zeros(batch_size, device=device) # [B]
    for d in range(model_module.length):
        logits = model(x)[:, :, :-1] # [B, D, N-1]
        update = sample_categorical_logits(logits[batch_arange, jump_pos[:, d]]) # [B]
        if torch.is_grad_enabled(): # avoid issues with in-place operations
            x = x.clone()
        x[batch_arange, jump_pos[:, d]] = update
        log_rnd_running += -np.log(model_module.vocab_size-1) - logits[batch_arange, jump_pos[:, d], update]
    rwd = reward_model(x) # [B]
    return x, log_rnd_running, rwd


@torch.no_grad()
def sampling_in_rounds(model, reward_model, batch_size, rounds=1, use_tqdm=False, device='cuda:0'):
    all_samples = []; all_log_rnd_runnings = []; all_rewards = []
    pbar = range(rounds) if not use_tqdm else tqdm(range(rounds), leave=False, desc='Sampling')
    for _ in pbar:
        x, log_rnd_running, rwd = rnd_mask(model, reward_model, batch_size, device=device)
        all_samples.append(x); all_log_rnd_runnings.append(log_rnd_running); all_rewards.append(rwd)
    return torch.cat(all_samples), torch.cat(all_log_rnd_runnings), torch.cat(all_rewards)


def loss_wdce_mask_lambda(model, log_rnd, x, num_replicates=16, antithetic=True):
    r"""Weighted denoising cross entropy loss in lambda formulation.
    X_T ~ P^\theta_T and weights \log\frac{dP^*}{dP^\theta}(X)
    
    log_rnd: [B]; x: [B, D] (clean data without mask state)
    num_replicates: R, number of replicates of each row in x
    
    antithetic: if True, use antithetic sampling to reduce variance
    i.e., for each sample x, sample num_replicates // 2 pairs of masked samples that are complementary
    """
    model_module = model.module if hasattr(model, 'module') else model # for DDP
    D = x.shape[1]
    batch = x.repeat([num_replicates, 1]) # [B*R, D]
    weights = log_rnd.softmax(dim=-1)
    batch_weights = weights.repeat([num_replicates]) # [B*R]
    if not antithetic:
        lamda = torch.rand(batch.shape[0], device=x.device) # [B*R]
        masked_index = torch.rand(*batch.shape, device=x.device) < lamda[..., None] # [B*R, D]
    else:
        lamda = torch.rand(batch.shape[0] // 2, device=x.device) # [B*R/2]
        masked_index = torch.rand([batch.shape[0] // 2, batch.shape[1]], device=x.device) < lamda[..., None] # [B*R/2, D]
        lamda = torch.cat([lamda, 1 - lamda], dim=0) # [B*R]
        masked_index = torch.cat([masked_index, ~masked_index], dim=0) # [B*R, D]
    perturbed_batch = torch.where(masked_index, model_module.vocab_size-1, batch)
    logits = model(perturbed_batch)
    losses = torch.zeros(*batch.shape, device=x.device, dtype=logits.dtype) # [B*R, D]
    losses[masked_index] = torch.gather(input=logits[masked_index], dim=-1,
                                        index=batch[masked_index][..., None]).squeeze(-1)
    lamda_weights = 1 / lamda / D # [B*R], theoretically should be 1 / lamda, we divide by D for smaller loss scales
    return - (losses.sum(dim=-1) * lamda_weights * batch_weights).sum() / num_replicates


def sample_masked_index(m, D):
    """
    m: [bsz], values in [1, D]
    Returns a boolean mask of shape [bsz, D]: in the b-th row, the number of True is exactly m[b].
    """
    bsz = m.shape[0]
    sorted_indices = torch.rand(bsz, D, device=m.device).argsort(dim=1)
    # [bsz, D], each row is random permutation of range(D)
    mask = torch.arange(D, device=m.device).expand(bsz, D) < m.unsqueeze(1)
    # [bsz, D], the b-th row is True for the first m[b] elements
    masked_index = torch.zeros(bsz, D, dtype=torch.bool, device=m.device)
    masked_index.scatter_(dim=1, src=mask, index=sorted_indices)
    return masked_index
    
def loss_wdce_mask_m(model, log_rnd, x, num_replicates=16, antithetic=True):
    r"""Weighted denoising cross entropy loss in m formulation.
    X_T ~ P^\theta_T and weights \log\frac{dP^*}{dP^\theta}(X)
    
    log_rnd: [B]; x: [B, D] (clean data without mask state)
    num_replicates: R, number of replicates of each row in x
    
    antithetic: if True, use antithetic sampling to reduce variance
    i.e., for each sample x, sample num_replicates // 2 pairs of masked samples that are complementary
    """

    model_module = model.module if hasattr(model, 'module') else model # for DDP
    D = x.shape[1]
    batch = x.repeat([num_replicates, 1]) # [B*R, D]
    weights = log_rnd.softmax(dim=-1)
    batch_weights = weights.repeat([num_replicates]) # [B*R]
    if not antithetic:
        m = torch.randint(1, D+1, (batch.shape[0],), device=x.device) # [B*R]
        masked_index = sample_masked_index(m, D) # [B*R, D]
    else:
        m = torch.randint(1, D+1, (batch.shape[0] // 2,), device=x.device) # [B*R/2]
        masked_index = sample_masked_index(m, D) # [B*R/2, D]
        m = torch.cat([m, D - m], dim=0) # [B*R]
        masked_index = torch.cat([masked_index, ~masked_index], dim=0) # [B*R, D]
    perturbed_batch = torch.where(masked_index, model_module.vocab_size-1, batch)
    logits = model(perturbed_batch)
    losses = torch.zeros(*batch.shape, device=x.device, dtype=logits.dtype) # [B*R, D]
    losses[masked_index] = torch.gather(input=logits[masked_index], dim=-1,
                                        index=batch[masked_index][..., None]).squeeze(-1)
    m_weights = 1 / m # [B*R], theoretically should be D / m, we remove D for smaller loss scales
    return - (losses.sum(dim=-1) * m_weights * batch_weights)[m!=0].sum() / num_replicates


def train_pdns(model, reward_fn, ema: EMA, args, logger: Logger):
    r"""Trainer for PDNS
    
    NOTE: The notation in the code is different from the paper (proposition 3.1),
    as listed in the table below for better readability.
    Please be careful when referring to the paper.
    
    | Code | Paper Notation | Description |
    | --- | --- | --- |
    | coeff(i) | \eta_i / (1 + \eta_i) | \eta_i is the proximal stepsize for the i-th stage |
    | mu(i) | \lambda_i | \lambda_i is the mixture coefficient for the i-th stage |

         mu(k) = (1 - coeff(1)) ... (1 - coeff(k))   [Code]
    <==> \lambda_k = \prod_{i=1}^k 1 / (1 + \eta_i)  [Paper eq. (9)]
    """

    assert len(args.pdns_coeffs) == args.pdns_outer_loops == len(args.pdns_inner_loops)
    if args.get('pdns_mus') is None: # compute mu given coeff
        args.pdns_mus = get_mu_from_coeff(args.pdns_coeffs) # mu(k) = (1 - coeff(1)) ... (1 - coeff(k))
    else:
        assert len(args.pdns_mus) == args.pdns_outer_loops
    update_count = sum(args.pdns_inner_loops[:args.start_step_stage - 1]) if args.pdns_inner_loops is not None else 0
    # num of gradient updates performed, if start from scratch (args.start_step_stage = 1) then is 0

    loss_fn = {'lambda': partial(loss_wdce_mask_lambda, antithetic=args.wdce_antithetic),
                    'm': partial(loss_wdce_mask_m, antithetic=args.wdce_antithetic)
               }.get(args.wdce_loss_formulation)

    model_module = model.module if hasattr(model, 'module') else model # for DDP

    for stage in range(args.start_step_stage, args.pdns_outer_loops+1):
        # k = stage, P^{\theta_{k-1}} -> P^k or P^{\theta_k^*}
        cur_coeff = args.pdns_coeffs[stage-1]
        cur_mu = args.pdns_mus[stage-1]
        logger.info(f">>> Stage k = {stage} / {args.pdns_outer_loops}, coeff(k) = {cur_coeff:.4f}, mu(k) = {cur_mu:.4f}")

        # EMA reinitialization
        ema.reset() # reset the number of updates as 0
        ema.copy_to(model.parameters()) # now the model parameters are the EMA parameters of the previous model P^{\theta_{k-1}}
        ema.store(model.parameters(), idx=1) # save a copy for generating the buffer

        # initialize the optimizer at the first stage;
        # reset the optimizer if specified
        if stage == args.start_step_stage or (
            args.optimizer.reset_every > 0 and (stage - 1) % args.optimizer.reset_every == 0):
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.optimizer.lr, weight_decay=args.optimizer.weight_decay)
            logger.info(f"Stage k = {stage} / {args.pdns_outer_loops}, optimizer (re)initialized.")

        # inner loop for training the model to P^k or P^{\theta_k^*}
        for step in range(1, args.pdns_inner_loops[stage-1]+1):

            if (step - 1) % args.sample_buffer_every == 0:
                with torch.no_grad():
                    ema.store(model.parameters())
                    if not args.pdns_on_policy: # sampling from the previous (EMA) model P^{\theta_{k-1}}
                        ema.restore(model.parameters(), idx=1)
                    else: # sampling from the current (EMA) model P^\theta, TODO: may be problematic for v2
                        ema.copy_to(model.parameters())
                    x_buffer, log_rnd_running_buffer, reward_buffer = rnd_mask(model, reward_fn, args.buffer_size, device=args.device)
                    ema.restore(model.parameters())
                    logger.info(f"Drawn {x_buffer.shape[0]} samples for the buffer.")

                if args.pdns_loss_version == 'v1': # log dP^k / dP^{\theta_{k-1}}
                    log_rnd_buffer = log_rnd_running_buffer + (1 - cur_mu) * reward_buffer
                else: # log dP^{\theta^*_k} / dP^{\theta_{k-1}}
                    log_rnd_buffer = cur_coeff * (log_rnd_running_buffer + reward_buffer)

                if step == 1: # at the first step, also visualize and evaluate the samples at last stage P^{\theta_{k-1}}
                    if args.dist == 'ising':
                        logger.log({'fig_samples': ising2d_visualize(x_buffer[:64]*2-1, num_per_row=16), 
                                    'fig_2pt_corr': ising2d_plot_2pt_corr(x_buffer*2-1)},
                                    step=update_count)
                    elif args.dist == 'potts':
                        logger.log({'fig_samples': potts2d_visualize(x_buffer[:64], num_per_row=16, q=args.tokens),
                                    'fig_2pt_corr': potts2d_plot_2pt_corr(x_buffer, q=args.tokens)},
                                    step=update_count)

            model.train(); optimizer.zero_grad()

            if not args.wdce_imp_samp_buffer:
                # uniformly sample from the buffer with weights and use them for computing loss
                idx = np.random.choice(args.buffer_size, args.batch_size, replace=False)
                x = x_buffer[idx]; log_rnd = log_rnd_buffer[idx]
            else:
                # sample from the buffer according to the weights and treat them as uniform samples for computing loss
                idx = torch.multinomial(log_rnd_buffer.softmax(dim=-1), args.batch_size, replacement=True)
                x = x_buffer[idx]; log_rnd = torch.zeros(args.batch_size, device=x.device)

            loss = loss_fn(model, log_rnd, x, num_replicates=args.wdce_num_replicates)
            if args.world_size > 1:
                with torch.no_grad():
                    dist.all_reduce(loss); loss /= args.world_size
            loss.backward(); update_count += 1
            info = {'mu': cur_mu, 'loss': loss.item(), 
                    'ess_train': ess(log_rnd) if not args.wdce_imp_samp_buffer else ess(log_rnd_buffer)}

            if args.optimizer.grad_clip:
                total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.optimizer.grad_clip_norm)
                info['total_grad_norm'] = total_grad_norm.item()
            elif step % args.optimizer.log_grad_norm_every == 0:
                grads = [p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]
                total_grad_norm = torch.cat(grads).norm()
                info['total_grad_norm'] = total_grad_norm.item()
            optimizer.step(); ema.update(model.parameters())

            if step % args.eval_every == 0: # evaluate ess on the ema model
                model.eval()
                with torch.no_grad():
                    ema.store(model.parameters())
                    ema.copy_to(model.parameters())
                    x, log_rnd_running, rwd = rnd_mask(model, reward_fn, args.eval_batch_size, device=args.device)
                    x, log_rnd_running, rwd = map(
                        lambda tensor: gather_across_processes(tensor, world_size=args.world_size, device=args.device),
                        (x, log_rnd_running, rwd)
                    )
                    ema.restore(model.parameters())

                    info['reward'] = rwd.mean().item() # E_{P^\theta} r(X)
                    
                    log_rnd_global = log_rnd_running + rwd # W^\theta(X)
                    info['ess_eval_global'] = ess(log_rnd_global) # ESS w.r.t. P^*
                    info['elbo_global'] = log_rnd_global.mean().item()
                    # E_{P^\theta} W^\theta(X) = log Z - KL(P^\theta || P^*)
                    info['log_z_global'] = torch.logsumexp(log_rnd_global, dim=0).item() - np.log(log_rnd_global.shape[0])
                    # log Z = log E_{P^\theta} exp(W^\theta(X))
                    info['path_kl_global'] = info['log_z_global'] - info['elbo_global']
                    # KL(P^\theta || P^*)

                    log_rnd_local = log_rnd_running + (1 - cur_mu) * rwd # W^\theta_k(X)
                    info['ess_eval_local'] = ess(log_rnd_local) # ESS w.r.t. P^k
                    info['elbo_local'] = log_rnd_local.mean().item()
                    # E_{P^\theta} W^\theta_k(X) = log Z_k - KL(P^\theta || P^k)
                    info['log_z_local'] = torch.logsumexp(log_rnd_local, dim=0).item() - np.log(log_rnd_local.shape[0])
                    # log Z_k = log E_{P^\theta} exp(W^\theta_k(X))
                    info['path_kl_local'] = info['log_z_local'] - info['elbo_local']
                    # KL(P^\theta || P^k)
                    
                    if args.dist == 'ising':
                        info['mag'] = ising2d_mag(x*2-1)
                    elif args.dist == 'potts':
                        info['mag'] = potts2d_mag(x, q=args.tokens)

            logger.log(info, step=update_count)

        if args.device == 'cuda:0' and (stage % args.save_every_stages == 0 or stage == args.pdns_outer_loops):
            save_dir = os.path.join(args.logging.dir, f'ckpt{stage}.pth')
            torch.save({
                'model_state_dict': model_module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ema_state_dict': ema.state_dict(),
                'cfg': args},
                save_dir)
            logger.info(f'Checkpoint saved at {save_dir}')

    # visualize final samples (with a smaller batch size than the buffer size)
    with torch.no_grad():
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        x, _, _ = rnd_mask(model, reward_fn, args.eval_batch_size, device=args.device)
        ema.restore(model.parameters())
    if args.dist == 'ising':
        logger.log({'fig_samples': ising2d_visualize(x[:64]*2-1, num_per_row=16), 
                    'fig_2pt_corr': ising2d_plot_2pt_corr(x*2-1)},
                    step=update_count)
    elif args.dist == 'potts':
        logger.log({'fig_samples': potts2d_visualize(x[:64], num_per_row=16, q=args.tokens),
                    'fig_2pt_corr': potts2d_plot_2pt_corr(x, q=args.tokens)},
                    step=update_count)


def train_pdns_adaptive(model, reward_fn, ema: EMA, args, logger: Logger):
    """
    Trainer for PDNS with adaptive stepsize based on ESS
    NOTE: Please use with caution as it hasn't been adapted for resume training from ckpt or DDP
    """

    cur_mu = 1 # mu(k) = (1 - coeff(1)) ... (1 - coeff(k))
    update_count = 0
    # num of gradient updates performed
    loss_fn = {'lambda': partial(loss_wdce_mask_lambda, antithetic=args.wdce_antithetic),
                    'm': partial(loss_wdce_mask_m, antithetic=args.wdce_antithetic)
               }.get(args.wdce_loss_formulation)
    refine_count_down = args.pdns_ada_refine_stages

    model_module = model.module if hasattr(model, 'module') else model # for DDP

    for stage in range(args.start_step_stage, args.pdns_ada_max_outer_loops + 1):
        # k = stage, P^{\theta_{k-1}} -> P^k or P^{\theta_k^*}

        if cur_mu == 0: # refine for at most pdns_ada_refine_stages stages
            refine_count_down -= 1
            if refine_count_down == 0: break

        # EMA reinitialization
        ema.reset() # reset the number of updates as 0
        ema.copy_to(model.parameters()) # now the model parameters are the EMA parameters of the previous model P^{\theta_{k-1}}
        ema.store(model.parameters(), idx=1) # save a copy for generating the buffer

        # initialize the optimizer at the first stage;
        # reset the optimizer for if specified, but not in the refinement stages
        if stage == 1 or (
            (args.optimizer.reset_every > 0 and (stage - 1) % args.optimizer.reset_every == 0)
            and refine_count_down == args.pdns_ada_refine_stages):
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.optimizer.lr, weight_decay=args.optimizer.weight_decay)

        # inner loop for training the model to P^k or P^{\theta_k^*}
        for step in range(1, args.pdns_ada_max_inner_loops):

            if (step - 1) % args.sample_buffer_every == 0:
                # sample buffer from the model P^{\theta_{k-1}} or P^\theta

                with torch.no_grad():
                    ema.store(model.parameters())
                    if not args.pdns_on_policy: # sampling from the previous (EMA) model P^{\theta_{k-1}}
                        ema.restore(model.parameters(), idx=1)
                    else: # sampling from the current (EMA) model P^\theta, TODO: may be problematic for v2
                        ema.copy_to(model.parameters())
                    x_buffer, log_rnd_running_buffer, reward_buffer = rnd_mask(model, reward_fn, args.buffer_size, device=args.device)
                    ema.restore(model.parameters())
                    logger.info(f"Drawn {x_buffer.shape[0]} samples for the buffer.")

                # at the first step, adaptively decide coeff based on ESS
                if step == 1:

                    if stage == args.pdns_ada_max_outer_loops or refine_count_down < args.pdns_ada_refine_stages:
                        # reaching max stages or in refinement stages, use infinite step size eta
                        cur_coeff = 1; cur_mu = 0
                    else: # use the buffer to decide the step size for this stage
                        cur_coeff = find_coeff_kl(log_rnd_running_buffer, reward_buffer, cur_mu,
                                                  target_kl=args.pdns_ada_kl_threshold,
                                                  pdns_loss_version=args.pdns_loss_version,
                                                  logger=logger)
                        cur_mu *= 1 - cur_coeff
                        if cur_mu < args.pdns_ada_term_mu:
                            # close to target distribution, use infinite step size eta
                            cur_coeff = 1; cur_mu = 0 # will cause the loop to terminate after refinement stages

                    logger.info(f">>> Stage k = {stage}, coeff(k) = {cur_coeff:.4f}, mu(k) = {cur_mu:.4f}")

                ## the following is copied from train_pdns without change ##

                if args.pdns_loss_version == 'v1': # log dP^k / dP^{\theta_{k-1}}
                    log_rnd_buffer = log_rnd_running_buffer + (1 - cur_mu) * reward_buffer
                else: # log dP^{\theta^*_k} / dP^{\theta_{k-1}}
                    log_rnd_buffer = cur_coeff * (log_rnd_running_buffer + reward_buffer)

                if step == 1: # at the first step, also visualize and evaluate the samples at last stage P^{\theta_{k-1}}
                    if args.dist == 'ising':
                        logger.log({'fig_samples': ising2d_visualize(x_buffer[:64]*2-1, num_per_row=16), 
                                    'fig_2pt_corr': ising2d_plot_2pt_corr(x_buffer*2-1)},
                                    step=update_count)
                    elif args.dist == 'potts':
                        logger.log({'fig_samples': potts2d_visualize(x_buffer[:64], num_per_row=16, q=args.tokens),
                                    'fig_2pt_corr': potts2d_plot_2pt_corr(x_buffer, q=args.tokens)},
                                    step=update_count)

            model.train(); optimizer.zero_grad()

            if not args.wdce_imp_samp_buffer:
                # uniformly sample from the buffer with weights and use them for computing loss
                idx = np.random.choice(args.buffer_size, args.batch_size, replace=False)
                x = x_buffer[idx]; log_rnd = log_rnd_buffer[idx]
            else:
                # sample from the buffer according to the weights and treat them as uniform samples for computing loss
                idx = torch.multinomial(log_rnd_buffer.softmax(dim=-1), args.batch_size, replacement=True)
                x = x_buffer[idx]; log_rnd = torch.zeros(args.batch_size, device=x.device)

            loss = loss_fn(model, log_rnd, x, num_replicates=args.wdce_num_replicates)
            if args.world_size > 1:
                with torch.no_grad():
                    dist.all_reduce(loss); loss /= args.world_size
            loss.backward(); update_count += 1
            info = {'mu': cur_mu, 'coeff': cur_coeff, 'loss': loss.item(), 
                    'ess_train': ess(log_rnd) if not args.wdce_imp_samp_buffer else ess(log_rnd_buffer)}

            if args.optimizer.grad_clip:
                total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.optimizer.grad_clip_norm)
                info['total_grad_norm'] = total_grad_norm.item()
            elif step % args.optimizer.log_grad_norm_every == 0:
                grads = [p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]
                total_grad_norm = torch.cat(grads).norm()
                info['total_grad_norm'] = total_grad_norm.item()
            optimizer.step(); ema.update(model.parameters())

            if step % args.eval_every == 0: # evaluate ess on the ema model
                model.eval()
                with torch.no_grad():
                    ema.store(model.parameters())
                    ema.copy_to(model.parameters())
                    x, log_rnd_running, rwd = rnd_mask(model, reward_fn, args.eval_batch_size, device=args.device)
                    ema.restore(model.parameters())

                    info['reward'] = rwd.mean().item() # E_{P^\theta} r(X)
                    
                    log_rnd_global = log_rnd_running + rwd # W^\theta(X)
                    info['ess_eval_global'] = ess(log_rnd_global) # ESS w.r.t. P^*
                    info['elbo_global'] = log_rnd_global.mean().item()
                    # E_{P^\theta} W^\theta(X) = log Z - KL(P^\theta || P^*)
                    info['log_z_global'] = torch.logsumexp(log_rnd_global, dim=0).item() - np.log(log_rnd_global.shape[0])
                    # log Z = log E_{P^\theta} exp(W^\theta(X))
                    info['path_kl_global'] = info['log_z_global'] - info['elbo_global']
                    # KL(P^\theta || P^*)

                    log_rnd_local = log_rnd_running + (1 - cur_mu) * rwd # W^\theta_k(X)
                    info['ess_eval_local'] = ess(log_rnd_local) # ESS w.r.t. P^k
                    info['elbo_local'] = log_rnd_local.mean().item()
                    # E_{P^\theta} W^\theta_k(X) = log Z_k - KL(P^\theta || P^k)
                    info['log_z_local'] = torch.logsumexp(log_rnd_local, dim=0).item() - np.log(log_rnd_local.shape[0])
                    # log Z_k = log E_{P^\theta} exp(W^\theta_k(X))
                    info['path_kl_local'] = info['log_z_local'] - info['elbo_local']
                    # KL(P^\theta || P^k)
                    
                    if args.dist == 'ising':
                        info['mag'] = ising2d_mag(x*2-1)
                    elif args.dist == 'potts':
                        info['mag'] = potts2d_mag(x, q=args.tokens)

            logger.log(info, step=update_count)

            ## move to the next stage when ess_eval_local is good enough ##
            if (info.get('ess_eval_local', 0) > args.pdns_ada_term_ess 
                and cur_mu > 0 and step >= args.pdns_ada_min_inner_loops):
                break

            ## in refinement stages, the max number of inner loops is pdns_ada_refine_inner_loops ##
            if refine_count_down < args.pdns_ada_refine_stages and step == args.pdns_ada_refine_inner_loops:
                break


        if args.device == 'cuda:0' and (stage % args.save_every_stages == 0 or refine_count_down == 1):
            save_dir = os.path.join(args.logging.dir, f'ckpt{stage}.pth')
            torch.save({
                'model_state_dict': model_module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ema_state_dict': ema.state_dict(),
                'cfg': args},
                save_dir)
            logger.info(f'Checkpoint saved at {save_dir}')

    # visualize final samples (with a smaller batch size than the buffer size)
    with torch.no_grad():
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        x, _, _ = rnd_mask(model, reward_fn, args.eval_batch_size, device=args.device)
        ema.restore(model.parameters())
    if args.dist == 'ising':
        logger.log({'fig_samples': ising2d_visualize(x[:64]*2-1, num_per_row=16), 
                    'fig_2pt_corr': ising2d_plot_2pt_corr(x*2-1)},
                    step=update_count)
    elif args.dist == 'potts':
        logger.log({'fig_samples': potts2d_visualize(x[:64], num_per_row=16, q=args.tokens),
                    'fig_2pt_corr': potts2d_plot_2pt_corr(x, q=args.tokens)},
                    step=update_count)
