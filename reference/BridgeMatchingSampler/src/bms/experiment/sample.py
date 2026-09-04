# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Generate samples from a trained BMS checkpoint and evaluate them with boltzkit.

Example
-------
    python -u -m bms.experiment.sample \
        experiment=ala2 \
        checkpoint_directory=ckpts/ala2/checkpoints \
        num_samples=100000
"""

from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm

from boltzkit.evaluation import EvalData, run_eval
from boltzkit.evaluation.eval import EnergyHistEval, make_wandb_compatible
from boltzkit.evaluation.molecular_eval import (
    DihedralAngleEval,
    TicaEval,
    TorsionMarginalEval,
)
from boltzkit.targets.boltzmann import MolecularBoltzmann

from bms.process.sde import ControlledSDE
from bms.utils.topology import save_data_to_pdb


def _latest_checkpoint(directory: Path) -> Path:
    checkpoints = list(directory.glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints (*.pt) found in {directory}")
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def _normalize_state_dict(state_dict: dict) -> dict:
    """Map a saved controller ``state_dict`` onto a plain ``ClippedModel``.

    Training may wrap the controller in ``torch.compile`` (``_orig_mod.`` prefix)
    and/or an ``EMA`` module. For EMA we keep the averaged ("shadow") weights.
    """
    # Strip the torch.compile wrapper prefix first, if present.
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    # Then prefer the EMA-averaged ("shadow") weights when present.
    if any(k.startswith("shadow.") for k in state_dict):
        state_dict = {
            k[len("shadow."):]: v
            for k, v in state_dict.items()
            if k.startswith("shadow.")
        }
    return state_dict


@torch.no_grad()
def generate(cfg: DictConfig, device: torch.device):
    controller = hydra.utils.instantiate(cfg.controller).to(device)

    checkpoint_path = _latest_checkpoint(Path(cfg.checkpoint_directory))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["controller"]
    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()
    controller.load_state_dict(_normalize_state_dict(state_dict))
    controller.eval()

    source = hydra.utils.instantiate(cfg.source).to(device)
    base_sde = hydra.utils.instantiate(cfg.base_sde).to(device)
    sde = ControlledSDE(base_sde=base_sde, controller=controller)
    integrator = hydra.utils.instantiate(cfg.integrator, sde=sde).to(device)

    num_samples = int(cfg.num_samples)
    batch_size = int(cfg.inference_batch_size)
    num_batches = (num_samples + batch_size - 1) // batch_size
    samples = []
    for _ in tqdm(range(num_batches), desc="sampling"):
        data_0 = source.sample(batch_size).to(device)
        data_1 = integrator.run(
            initial_data=data_0,
            center_every_step=cfg.mean_free,
            zero_last_step_noise=False,
            return_trajectory=False,
            progress_bar=False,
        )
        samples.append(data_1.cpu())
    return torch.cat(samples, dim=0)[:num_samples]


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = generate(cfg, device)  # (num_samples, n_atoms, 3), Angstrom

    out_dir = Path(HydraConfig.get().run.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"pos": samples}, out_dir / "samples.pt")
    if hasattr(cfg, "topology_pdb_file"):
        save_data_to_pdb(
            data=samples.numpy(),
            topology_file=cfg.topology_pdb_file,
            output_file=str(out_dir / "samples.pdb"),
        )

    # Evaluate against boltzkit reference data.
    system: MolecularBoltzmann = hydra.utils.instantiate(
        cfg.potential.potentials[0].system
    )
    val_data = system.load_dataset(T=cfg.temperature, type="val").get_samples()

    topology = system.get_mdtraj_topology()
    eval_pipeline = [
        EnergyHistEval(),
        TorsionMarginalEval(topology),
        TicaEval(topology, system.get_tica_model()),
        DihedralAngleEval(topology, system.get_z_matrix()),
    ]

    xyz = samples.numpy()
    pred_flat = xyz.reshape(xyz.shape[0], -1)
    min_size = min(pred_flat.shape[0], val_data.shape[0])
    eval_data = EvalData(
        true_samples_target_log_prob=system.get_log_prob(val_data[:min_size]),
        pred_samples_target_log_prob=system.get_log_prob(pred_flat[:min_size]),
        samples_true=val_data[:min_size],
        samples_pred=pred_flat[:min_size],
    )
    metrics = run_eval(data=eval_data, evals=eval_pipeline)
    scalar_metrics = {
        k: float(v)
        for k, v in make_wandb_compatible(metrics, dpi=100).items()
        if isinstance(v, (int, float))
    }
    print("Evaluation metrics:")
    for key, value in scalar_metrics.items():
        print(f"  {key}: {value:.5f}")


if __name__ == "__main__":
    main()
