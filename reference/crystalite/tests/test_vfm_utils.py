import torch

from src.crystalite.vfm_utils import (
    compute_cartesian_vfm_loss,
    cartesian_vfm_edm_heun_sampler,
    endpoint_velocity,
    compute_vfm_loss,
    linear_interpolant,
    sample_packora_time,
    torus_delta,
    vfm_sampler,
    weighted_endpoint_l1,
)


class _CartesianEndpoint(torch.nn.Module):
    def forward(self, type_features, coords, lattice, pad_mask, t, **kwargs):
        return {"coord_vel": torch.zeros_like(coords), "lattice_vel": torch.zeros_like(lattice)}


class _FixedEndpoint(torch.nn.Module):
    def __init__(self, coord, lattice):
        super().__init__()
        self.coord, self.lattice = coord, lattice

    def forward(self, type_features, coords, lattice, pad_mask, t, **kwargs):
        return {
            "coord_vel": self.coord.expand_as(coords),
            "lattice_vel": self.lattice.expand_as(lattice),
        }


def test_cartesian_vfm_is_finite_and_com_centered():
    batch, atoms = 2, 4
    pad = torch.zeros(batch, atoms, dtype=torch.bool)
    frac = torch.rand(batch, atoms, 3)
    lattice = torch.zeros(batch, 6)
    result = compute_cartesian_vfm_loss(
        _CartesianEndpoint(), torch.zeros(batch, atoms, 2), frac, lattice, pad,
        torch.tensor([0.2, 0.8]), torch.ones(3), torch.zeros(6), torch.ones(6),
        augment_translation=False,
    )
    assert torch.isfinite(result["loss_total"])
    assert torch.allclose(result["clean_coord"].mean(1), torch.zeros(batch, 3), atol=1e-6)


def test_cartesian_clean_target_is_direct_cartesian_then_com_centered():
    frac = torch.tensor([[[0.1, 0.2, 0.3], [0.9, 0.4, 0.7]]])
    pad = torch.zeros(1, 2, dtype=torch.bool)
    result = compute_cartesian_vfm_loss(
        _CartesianEndpoint(), torch.zeros(1, 2, 2), frac, torch.zeros(1, 6), pad,
        torch.tensor([0.5]), torch.ones(3), torch.zeros(6), torch.ones(6),
        augment_translation=False,
    )
    expected = frac - frac.mean(1, keepdim=True)
    assert torch.allclose(result["clean_coord"], expected, atol=1e-7)


def test_torus_interpolant_and_velocity_follow_shortest_path():
    x0 = torch.tensor([[[0.9, 0.2, 0.3]]])
    x1 = torch.tensor([[[0.1, 0.4, 0.2]]])
    l0 = torch.zeros(1, 6)
    l1 = torch.ones(1, 6)
    t = torch.tensor([0.5])
    xt, lt = linear_interpolant(x0, x1, l0, l1, t)
    assert torch.allclose(xt, torch.tensor([[[0.0, 0.3, 0.25]]]), atol=1e-6)
    assert torch.allclose(lt, torch.full((1, 6), 0.5))
    vx, vl = endpoint_velocity(x1, xt, l1, lt, t)
    assert torch.allclose(vx, torus_delta(x1, x0), atol=1e-6)
    assert torch.allclose(vl, torch.ones(1, 6))


def test_weighted_l1_uses_torus_and_ignores_padding():
    clean = torch.tensor([[[0.99, 0.0, 0.0], [0.2, 0.2, 0.2]]])
    pred = torch.tensor([[[0.01, 0.0, 0.0], [0.9, 0.9, 0.9]]])
    losses = weighted_endpoint_l1(
        pred, clean, torch.ones(1, 6), torch.zeros(1, 6), torch.tensor([[False, True]])
    )
    assert torch.allclose(losses["loss_coord"], torch.tensor(0.02 / 3), atol=1e-7)
    assert torch.allclose(losses["loss_lattice"], torch.tensor(1.0))
    assert torch.allclose(losses["loss_total"], torch.tensor(10 * 0.02 / 3 + 1))


def test_packora_time_distribution_support_and_mean():
    torch.manual_seed(7)
    t = sample_packora_time(200_000, "cpu")
    assert bool(((0 <= t) & (t <= 1)).all())
    expected = 0.98 * (1.9 / 2.9) + 0.02 * 0.5
    assert abs(float(t.mean()) - expected) < 0.005


def test_vfm_batch_passes_temperature_and_uses_endpoint_outputs():
    class ExactEndpoint(torch.nn.Module):
        def __init__(self, frac, lattice):
            super().__init__()
            self.frac, self.lattice = frac, lattice

        def forward(self, *args, **kwargs):
            assert torch.allclose(kwargs["gem_sigma"], 1 - args[4])
            assert kwargs["temperature_present"].item()
            return {"coord_vel": self.frac, "lattice_vel": self.lattice}

    frac = torch.rand(1, 2, 3)
    lattice = torch.rand(1, 6)
    losses = compute_vfm_loss(
        ExactEndpoint(frac, lattice),
        torch.zeros(1, 2, 4),
        frac,
        lattice,
        torch.zeros(1, 2, dtype=torch.bool),
        torch.tensor([0.25]),
        temperature_k=torch.tensor([0.0]),
        temperature_present=torch.tensor([True]),
    )
    assert losses["loss_total"].item() == 0


def test_vfm_sampler_reaches_fixed_fractional_endpoint():
    pad = torch.zeros(2, 4, dtype=torch.bool)
    target_x = torch.full((1, 1, 3), 0.25)
    target_lattice = torch.full((1, 6), 0.5)
    out = vfm_sampler(
        _FixedEndpoint(target_x, target_lattice),
        torch.zeros(2, 4, 3),
        pad,
        num_steps=8,
        generator=torch.Generator().manual_seed(1),
    )
    assert torch.allclose(out["frac"], target_x.expand_as(out["frac"]), atol=1e-6)
    assert torch.allclose(out["lat"], target_lattice.expand_as(out["lat"]), atol=1e-6)


def test_cartesian_edm_heun_reaches_fixed_endpoint_without_churn():
    pad = torch.zeros(2, 4, dtype=torch.bool)
    target_coord = torch.zeros(1, 1, 3)
    target_lattice = torch.full((1, 6), 0.5)
    out = cartesian_vfm_edm_heun_sampler(
        _FixedEndpoint(target_coord, target_lattice),
        torch.zeros(2, 4, 3), pad, num_steps=8,
        coord_std=torch.ones(3), lattice_mean=torch.zeros(6), lattice_std=torch.ones(6),
        s_churn=0.0, generator=torch.Generator().manual_seed(1),
    )
    assert torch.allclose(out["frac"], torch.zeros_like(out["frac"]), atol=1e-5)
    assert torch.allclose(out["lat"], target_lattice.expand_as(out["lat"]), atol=1e-5)
