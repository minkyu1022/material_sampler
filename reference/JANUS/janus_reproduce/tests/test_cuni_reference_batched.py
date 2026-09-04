import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from janus_reproduce.cuni_reference_batched import (
    ReferenceConfig,
    ReferenceState,
    reference_states_108,
    run_reference_batch,
    species_log_acceptance,
    volume_log_acceptance,
)


class ToyOracle(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("anchor", torch.tensor(0.0, dtype=torch.float64))
        self.path = Path("toy.eam")

    def forward(self, species, fractional, log_volume):
        return fractional.square().sum((1, 2)) + 0.02 * species.sum(1) + log_volume.square()


def test_published_grid_and_opposing_walkers_are_exact():
    states = reference_states_108()
    assert len(states) == 15 * 33 * 2 == 990
    assert len({state.temperature for state in states}) == 15
    assert len({state.delta_mu for state in states}) == 33
    assert [(states[0].walker, states[1].walker), (states[-2].walker, states[-1].walker)] == [
        (0, 1),
        (0, 1),
    ]


def test_semigrand_acceptance_has_log_volume_jacobian_and_cu_mu_sign():
    beta = torch.tensor([2.0])
    assert volume_log_acceptance(
        beta,
        torch.tensor([3.0]),
        torch.tensor([5.0]),
        torch.tensor([0.1]),
        pressure=0.2,
        n_atoms=108,
    ).item() == pytest.approx(-2 * (3 + 0.2 * 5) + 108 * 0.1)
    # Positive delta_mu favors Ni->Cu (delta N_Cu=+1), not the reverse.
    forward = species_log_acceptance(
        beta, torch.tensor([0.3]), torch.tensor([0.8]), torch.tensor([1])
    )
    reverse = species_log_acceptance(
        beta, torch.tensor([-0.3]), torch.tensor([0.8]), torch.tensor([-1])
    )
    assert forward.item() == pytest.approx(1.0)
    assert reverse.item() == pytest.approx(-1.0)


def test_short_batch_writes_raw_traces_thinned_configs_and_production_stats(tmp_path):
    states = [ReferenceState(0, 500.0, 0.6, 0), ReferenceState(1, 500.0, 0.6, 1)]
    config = ReferenceConfig(
        total_sweeps=5,
        burn_in=2,
        config_thin=2,
        species_moves=2,
        adapt_interval=1,
        checkpoint_interval=2,
        seed=7,
    )
    paths = run_reference_batch(ToyOracle(), states, tmp_path, config=config)

    assert len(paths) == 2
    for state, path in zip(states, paths, strict=True):
        result = np.load(path)
        metadata = json.loads(result["metadata"].item())
        assert result["energy_eV"].shape == (5,)
        assert result["n_cu"].shape == (5,)
        assert result["log_volume"].shape == (5,)
        assert result["config_sweeps"].tolist() == [2, 4]
        assert result["fractional_positions"].shape == (2, 108, 3)
        assert result["initial_fractional_positions"].shape == (108, 3)
        assert result["species"].shape == (2, 108)
        assert set(np.unique(result["species"])) <= {0, 1}
        assert metadata["production_start"] == 2
        assert metadata["burn_in"] == 2
        assert metadata["total_sweeps"] == 5
        assert metadata["config_stride"] == 2
        assert metadata["initial_phase"] == ("all-Ni" if state.walker == 0 else "all-Cu")
        assert metadata["config"]["total_sweeps"] == 5
        assert set(metadata["production_acceptance"]) == {"displacement", "volume", "species"}
        assert all(0 <= value <= 1 for value in metadata["production_acceptance"].values())
        assert metadata["production_attempts"] == {
            "displacement": 3,
            "volume": 3,
            "species": 6,
        }
    assert not list(tmp_path.glob("checkpoint_*.pt"))


def test_config_rejects_burn_in_outside_total():
    with pytest.raises(ValueError, match="burn_in"):
        ReferenceConfig(total_sweeps=10, burn_in=10)
