import math
from pathlib import Path

import pytest
import torch

from janus_reproduce.cuni import KB_EV_K
from janus_reproduce.cuni_train import (
    CuNiTrainConfig,
    ReplayBuffer,
    _clip_field,
    _euler_maruyama,
    _label,
    _prior_values,
    _reference,
    train_cuni,
)
from janus_reproduce.torch_eam import TorchCuNiEAM

POTENTIAL = Path(__file__).parents[1] / "potentials/cu_ni/Cu_Ni_Fischer_2018.eam.alloy"


def test_paper_defaults_and_replay_limit():
    config = CuNiTrainConfig(POTENTIAL)
    assert (config.steps, config.initial_buffer, config.rounds) == (100, 5_000, 120)
    assert (config.fresh_per_round, config.replay_size, config.updates_per_round) == (
        1_000,
        5_000,
        500,
    )
    assert config.global_batch == 96
    assert (config.diffusion_u, config.diffusion_v) == (0.0, 0.0)
    buffer = ReplayBuffer()
    batch = {key: torch.arange(3)[:, None] for key in buffer.fields}
    buffer.add(batch, 2)
    assert len(buffer) == 2


def test_species_prior_is_all_mask_without_composition_logit_bias():
    config = CuNiTrainConfig.smoke(POTENTIAL, Path("unused"))
    prior = __import__("janus_reproduce.cuni", fromlist=["VolumePrior"]).VolumePrior(
        11.0, 12.0, 0.0, 0.0, 0.01
    )
    values = _prior_values(
        config, prior, torch.tensor([900.0]), torch.tensor([0.8]), config.n_atoms
    )
    assert len(values) == 4


def test_euler_maruyama_includes_score_and_noise_in_channel_coordinates():
    result = _euler_maruyama(
        torch.tensor([1.0]),
        velocity=torch.tensor([2.0]),
        score=torch.tensor([3.0]),
        diffusion=torch.tensor([0.5]),
        dt=0.2,
        noise=torch.tensor([4.0]),
    )
    expected = 1.0 + (2.0 + 0.5**2 * 3.0) * 0.2 + (2 * 0.5**2 * 0.2) ** 0.5 * 4.0
    assert result.item() == pytest.approx(expected)


def test_rollout_field_clipping_bounds_vectors_and_scalars():
    vector = _clip_field(torch.tensor([[[3.0, 4.0, 0.0]]]), 2.0, vector=True)
    scalar = _clip_field(torch.tensor([-3.0, 0.5, 4.0]), 1.0, vector=False)
    assert vector.norm(dim=-1).item() == pytest.approx(2.0)
    assert scalar.tolist() == [-1.0, 0.5, 1.0]


def test_cpu_smoke_and_resume(tmp_path):
    output = tmp_path / "run"
    config = CuNiTrainConfig.smoke(POTENTIAL, output)
    history = train_cuni(config)
    assert len(history) == 1
    assert torch.isfinite(torch.tensor(history[0]["loss"]))
    assert (output / "checkpoint.pt").is_file()

    resumed = train_cuni(CuNiTrainConfig.smoke(POTENTIAL, output, rounds=2))
    assert [row["round"] for row in resumed] == [1.0, 2.0]


def test_labels_match_semigrand_npt_log_density_finite_differences(tmp_path):
    config = CuNiTrainConfig.smoke(POTENTIAL, tmp_path, n_atoms=4, target_score_u_clip=None)
    oracle = TorchCuNiEAM(POTENTIAL)
    reference = _reference(4, torch.device("cpu")).double()
    species = torch.tensor([[0, 1, 0, 1]])
    displacement = torch.tensor(
        [
            [
                [0.003, -0.002, 0.001],
                [0.0, 0.002, -0.001],
                [-0.002, 0.0, 0.002],
                [0.001, 0.0, -0.002],
            ]
        ],
        dtype=torch.float64,
    )
    log_volume = torch.tensor([math.log(4 * 11.7)], dtype=torch.float64)
    temperature = torch.tensor([900.0], dtype=torch.float64)
    delta_mu = torch.tensor([0.84], dtype=torch.float64)
    states = {
        "species": species,
        "displacement": displacement,
        "log_volume": log_volume,
        "temperature": temperature,
        "delta_mu": delta_mu,
    }
    labels = _label(oracle, states, config, reference)
    beta = 1 / (KB_EV_K * temperature[0])

    def log_density(tokens, offsets, volume):
        energy = oracle(tokens, reference + offsets, volume)
        return -beta * energy + beta * delta_mu[0] * tokens.eq(0).sum() + config.n_atoms * volume

    epsilon = 1e-5
    plus, minus = displacement.clone(), displacement.clone()
    plus[0, 0, 0] += epsilon
    minus[0, 0, 0] -= epsilon
    finite_u = (
        log_density(species, plus, log_volume) - log_density(species, minus, log_volume)
    ) / (2 * epsilon)
    assert labels["score_u"][0, 0, 0].item() == pytest.approx(finite_u.item(), abs=2e-2)

    finite_v = (
        log_density(species, displacement, log_volume + epsilon)
        - log_density(species, displacement, log_volume - epsilon)
    ) / (2 * epsilon)
    assert labels["score_v"].item() == pytest.approx(finite_v.item(), abs=2e-3)

    alternatives = species.expand(2, -1).clone()
    alternatives[:, 0] = torch.tensor([0, 1])
    expected_heat_bath = torch.stack(
        [log_density(token[None], displacement, log_volume).squeeze() for token in alternatives]
    ).softmax(0)
    assert labels["heat_bath"][0, 0] == pytest.approx(expected_heat_bath, abs=2e-6)
