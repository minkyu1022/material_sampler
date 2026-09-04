import os
import numpy as np
import ot as pot
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
import torch
from mace.tools.torch_geometric import Batch

from src.components.sdes import BaseSDE, sdeint, loss_sdeint
from src.components.term_cost import TermCost
from src.components.scheduler import BaseScheduler
from src.utils.common import get_timesteps
from src.eval.distribution_distances import compute_distribution_distances, compute_distribution_distances2

import src.utils.plot_utils as plot_utils
import src.utils.chem_utils as chem_utils
import src.utils.chem_eval_utils as ceval_utils

from wandb import Image

from ipdb import set_trace as debug


def eval_epoch(
    fn: str,
    sde: BaseSDE,
    source: torch.nn.Module,
    energy: torch.nn.Module,
    term_cost: TermCost,
    gamma: BaseScheduler,
    beta: float,
    cfg: DictConfig,
    device:str = None,
):
    if cfg.chem:
        return chem_eval(fn, sde, source, energy, cfg, device=device)
    elif "AlanineDipeptideEnergy" in cfg.energy._target_:
        return ad_evaluation(fn, sde, source, energy, cfg, device=device)
    elif cfg.quick_evaluation:
        return quick_evaluation(fn, sde, source, energy, term_cost, gamma, beta, cfg, device=device)
    else:
        return toy_evaluation(fn, sde, source, energy, term_cost, gamma, beta, cfg, device=device)


def chem_eval(
    fn: str,
    sde: BaseSDE,
    eval_source: torch.nn.Module,
    energy: torch.nn.Module,
    cfg: DictConfig,
    device:str = None,
):
    # meta data
    atomic_numbers = energy.atomic_numbers

    # generate model samples
    graph_state0 = eval_source.sample([cfg.eval_batch_size,]).to(device)
    timesteps = get_timesteps(**cfg.timesteps).to(device)
    (graph_state0, graph_state1) = sdeint(
        sde,
        graph_state0,
        timesteps,
        zero_last_step_noise=cfg.zero_last_step_noise,
        only_boundary=True,
    )
    E_out = energy(graph_state1)

    # generate ref (CREST) samples
    conformers_file = to_absolute_path(cfg.conformers_file)
    graph_state_ref = chem_utils.load_crest_graph_state(
        conformers_file, energy
    )
    E_ref = energy(graph_state_ref.to(device), regularize=False)

    # eval
    results = {}
    os.makedirs("figs", exist_ok=True)
    gen_energy_img = ceval_utils.plot_atom_dist_and_energy(
        graph_state1,
        E_out,
        E_ref=E_ref,
        min_energy=cfg.min_energy,
        max_energy=cfg.max_energy,
    )
    gen_energy_img.save(f"figs/gen_energy_{fn}.png")
    # gen_conformer_img = ceval_utils.plot_conformer(
    #     graph_state1, E_out, atomic_numbers, n_samples=8
    # )
    # gen_conformer_img.save(f"figs/gen_conformer_{fn}.png")
    results["gen_energy_img"] = gen_energy_img

    init_energy_img = ceval_utils.plot_atom_dist_and_energy(
        graph_state0,
        energy(graph_state0),
        E_ref=E_ref,
        min_energy=cfg.min_energy,
        max_energy=cfg.max_energy,
    )
    init_energy_img.save(f"figs/init_energy.png")

    return results


def compute_recall_precision(
    gen_graph_state: Batch,
    energy: torch.nn.Module,
    cfg: DictConfig,
):
    # meta data: NOTE(ghliu) `eval_smiles` and `conformers_file` need to align
    atomic_numbers = energy.atomic_numbers
    charge = chem_utils.get_charge(cfg.eval_smiles)
    conformers_file = to_absolute_path(cfg.conformers_file)

    mol_out = chem_utils.generate_conformer(
        gen_graph_state, atomic_numbers, charge
    )
    mol_ref = chem_utils.load_crest_conformer(
        conformers_file, energy, charge
    )

    results = {}
    rmsd_metrics = ceval_utils.calculate_rmsd(
        mol_out, mol_ref, cfg.eval_smiles
    )
    results.update(rmsd_metrics)
    results["gen_recall_precision"] = ceval_utils.plot_recall_precision(
        rmsd_metrics, cfg.eval_smiles, label="gen",
    )
    # results["gen_recall_precision"].save(f"figs/gen_recall_precision_{fn}.png")
    return results


def ad_evaluation(
    fn: str,
    sde: BaseSDE,
    source: torch.nn.Module,
    energy: torch.nn.Module,
    cfg: DictConfig,
    log_steps: int = 5,
    device:str = None,
):

    # generate samples
    x0 = source.sample([cfg.eval_batch_size,]).to(device)
    timesteps = get_timesteps(**cfg.timesteps).to(x0)
    x0, x1 = sdeint(
        sde,
        x0,
        timesteps,
        zero_last_step_noise=cfg.zero_last_step_noise,
        only_boundary=True,
    )
    gen_samples = x1
    if hasattr(energy, "z_to_x"):
        # quick handle for ad_ic
        gen_samples = energy.z_to_x(gen_samples)

    # reference samples
    ref_samples_md = energy.sample_test_set(cfg.eval_batch_size)
    ref_samples = torch.from_numpy(ref_samples_md.xyz).reshape(-1, 66).to(device)
    assert gen_samples.shape == ref_samples.shape
    # energy W2
    if hasattr(energy, "eval_x"):
        # quick handle for ad_ic
        gen_energies = energy.eval_x(gen_samples).cpu()
        ref_energies = energy.eval_x(ref_samples).cpu()
    else:
        gen_energies = energy.eval(gen_samples).cpu()
        ref_energies = energy.eval(ref_samples).cpu()
    energy_w2 = pot.emd2_1d(ref_energies.numpy(), gen_energies.numpy())**0.5
    # print(energy_w2)

    # dist W2
    gen_int_dists = energy.interatomic_dist(gen_samples).cpu().numpy().reshape(-1)
    ref_int_dists = energy.interatomic_dist(ref_samples).cpu().numpy().reshape(-1)
    dist_w2 = pot.emd2_1d(gen_int_dists, ref_int_dists)

    eval_dict = {}
    eval_dict['energy_w2'] = energy_w2
    eval_dict['dist_w2'] = dist_w2

    from src.utils.md_utils import plot_marginal_group_AD, plot_energy_hist
    eval_dict.update(
        plot_marginal_group_AD(gen_samples.cpu(), ref_samples_md)
    )
    eval_dict.update(plot_energy_hist(gen_energies, ref_energies))

    return eval_dict


def toy_evaluation(
    fn: str,
    sde: BaseSDE,
    source: torch.nn.Module,
    energy: torch.nn.Module,
    term_cost: TermCost,
    gamma: BaseScheduler,
    beta: float,
    cfg: DictConfig,
    log_steps: int = 5,
    device:str = None,
):
    # generate samples
    x0 = source.sample([cfg.eval_batch_size,]).to(device)
    timesteps = get_timesteps(**cfg.timesteps).to(x0)
    (B, D), T = x0.shape, len(timesteps)

    # forward simulate
    xs, log_rnd = loss_sdeint(
        sde,
        x0,
        timesteps,
        zero_last_step_noise=cfg.zero_last_step_noise,
        only_boundary=False
    )

    # ESS
    log_rnd -= term_cost(xs[-1])
    weight = torch.softmax(log_rnd, dim=0)
    ESS = 1 / (weight**2).sum() / len(weight)
    weight_eta = torch.softmax(gamma() * log_rnd, dim=0)
    ESS_eta = 1 / (weight_eta**2).sum() / len(weight_eta)

    xs = torch.stack(xs)

    # subsample
    N = log_steps
    steps_to_log = np.linspace(0, T - 1, N).astype(int)
    ts = timesteps[steps_to_log]
    xs = xs[steps_to_log]
    assert ts.shape == (N,) and xs.shape == (N, B, D)
    generated_samples = xs[-1]

    # sample target
    if hasattr(energy, "sample"):
        global_target = target = energy.sample((B,))
        log_rnd = beta * term_cost(global_target)
        weight = torch.softmax(log_rnd, dim=0)
        indices = torch.multinomial(weight, len(weight), replacement=True)
        local_target = global_target[indices]

        assert global_target.shape == (B, D)
        eval_dict = compute_distribution_distances(
                energy.unnormalize(generated_samples),
                global_target.to(generated_samples.device),
                local_target.to(generated_samples.device),
                energy)
        eval_dict['global_ess'] = ESS
        eval_dict['train_ess'] = ESS_eta
    else:
        target = None

    if cfg.plot_lim:
        # plot
        plot_utils.plot_toy(
            ts,
            xs,
            target if target is not None else target,
            lim=cfg.plot_lim,
        )
        os.makedirs("figs", exist_ok=True)
        plot_utils.save_fig(fn)

    try: # is molecule (dw4, lj potentials)
        global_target = energy.sample_test_set(cfg.eval_batch_size)
        log_rnd = beta * term_cost(global_target)
        weight = torch.softmax(log_rnd, dim=0)
        indices = torch.multinomial(weight, len(weight), replacement=True)
        local_target = global_target[indices]

        generated_energies = energy.unnorm_log_prob(generated_samples)
        energies = energy.unnorm_log_prob(energy.normalize(global_target))
        local_energies = energy.unnorm_log_prob(energy.normalize(local_target))


        # get metrics
        energy_w2 = pot.emd2_1d(energies.cpu().numpy(), generated_energies.cpu().numpy())**0.5
        local_energy_w2 = pot.emd2_1d(local_energies.cpu().numpy(), generated_energies.cpu().numpy())**0.5
        # print(energy_w2)
        dist_w2 = pot.emd2_1d(
                energy.interatomic_dist(generated_samples).cpu().numpy().reshape(-1),
                energy.interatomic_dist(global_target).cpu().numpy().reshape(-1),
            )

        print('compute all eval metrics...')
        eval_dict = compute_distribution_distances2(
                energy.unnormalize(generated_samples),
                global_target.to(generated_samples.device),
                local_target.to(generated_samples.device),
                energy,
            )
        print('finished!')
        ##########################################################################################################
        eval_dict['energy_w2'] = energy_w2
        eval_dict['local_energy_w2'] = local_energy_w2
        eval_dict['dist_w2'] = dist_w2
        eval_dict['global_ess'] = ESS
        eval_dict['train_ess'] = ESS_eta
    except: pass

    return eval_dict


@torch.no_grad()
def simple_evaluation(
    epoch: int,
    stage: str,
    sde: BaseSDE,
    source: torch.nn.Module,
    energy: torch.nn.Module,
    cfg: DictConfig,
    log_steps: int = 5,
    device:str = None,
):
    x0 = source.sample([cfg.eval_batch_size,]).to(device)
    timesteps = get_timesteps(**cfg.timesteps).to(x0)
    (B, D), T = x0.shape, len(timesteps)
    # forward simulate
    xs = sdeint(
        sde,
        x0,
        timesteps,
        zero_last_step_noise=cfg.zero_last_step_noise,
        only_boundary=False,
    )
    xs = torch.stack(xs)

    # subsample
    N = log_steps
    steps_to_log = np.linspace(0, T - 1, N).astype(int)
    ts = timesteps[steps_to_log]
    xs = xs[steps_to_log]
    assert ts.shape == (N,) and xs.shape == (N, B, D)

    # sample target
    if hasattr(energy, "sample"):
        target = energy.sample((B,))
        assert target.shape == (B, D)
    else:
        target = None

    # plot
    plot_utils.plot_toy(
        ts,
        xs,
        target if target is not None else target,
        lim=cfg.plot_lim,
    )
    os.makedirs("figs", exist_ok=True)
    plot_utils.save_fig(f"ep{epoch}_{stage}")

@torch.no_grad()
def quick_evaluation(
    fn: str,
    sde: BaseSDE,
    source: torch.nn.Module,
    energy: torch.nn.Module,
    term_cost: TermCost,
    gamma: BaseScheduler,
    beta: float,
    cfg: DictConfig,
    log_steps: int = 5,
    device:str = None,
):
    # generate samples
    x0 = source.sample([cfg.eval_batch_size,]).to(device)
    timesteps = get_timesteps(**cfg.timesteps).to(x0)
    (B, D), T = x0.shape, len(timesteps)

    # forward simulate
    xs, log_rnd = loss_sdeint(
        sde,
        x0,
        timesteps,
        zero_last_step_noise=cfg.zero_last_step_noise,
        only_boundary=False
    )
    generated_samples = xs[-1]


    global_target = energy.sample_test_set(cfg.eval_batch_size)
    

    generated_energies = energy.unnorm_log_prob(generated_samples)
    energies = energy.unnorm_log_prob(energy.normalize(global_target))
    

    # get metrics
    energy_w2 = pot.emd2_1d(energies.cpu().numpy(), generated_energies.cpu().numpy())**0.5
    
    dist_w2 = pot.emd2_1d(
            energy.interatomic_dist(generated_samples).cpu().numpy().reshape(-1),
            energy.interatomic_dist(global_target).cpu().numpy().reshape(-1),
        )

    
    print('finished!')
    ##########################################################################################################
    eval_dict = {}
    eval_dict['energy_w2'] = energy_w2
    eval_dict['dist_w2'] = dist_w2

    return eval_dict