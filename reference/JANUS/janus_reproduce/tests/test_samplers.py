import pytest
import torch

from janus_reproduce.samplers import (
    DISCRETE_SAMPLER_REGISTRY,
    fixed_composition_boundary_quota,
    janus_tau_leap,
    sequential_random_order,
)


@pytest.mark.parametrize("target", (0, 1, 2, 4, 6, 7, 8))
def test_boundary_quota_reveals_once_and_hits_every_required_rung(target):
    sites = 8
    species = torch.full((1, sites), 2, dtype=torch.long)
    logits = torch.tensor([[[0.3, -0.2]]]).expand(1, sites, 2)
    order = torch.arange(sites)[None]
    seen = species.clone()
    logq = torch.zeros(1, dtype=torch.float64)
    for step in range(sites):
        result = fixed_composition_boundary_quota(
            species,
            logits,
            torch.tensor([target]),
            order[:, step],
            generator=torch.Generator().manual_seed(100 + step),
        )
        changed = result.species.ne(species)
        assert changed.sum() == 1
        assert changed[0, step]
        assert result.species[seen.ne(2)].eq(seen[seen.ne(2)]).all()
        species, seen = result.species, result.species.clone()
        logq += result.log_probability
    assert species.ne(2).all()
    assert species.eq(1).sum() == target
    assert torch.isfinite(logq).all()


def test_boundary_quota_forced_steps_add_zero_to_logq_and_match_manual_path():
    species = torch.full((1, 3), 2, dtype=torch.long)
    logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [-1.0, 1.0]]])
    target = torch.tensor([1])
    generator = torch.Generator().manual_seed(7)
    actual = torch.zeros(1, dtype=torch.float64)
    expected = torch.zeros(1, dtype=torch.float64)
    forced_logq = []
    for site in range(3):
        masked = species.eq(2).sum(1)
        remaining = target - species.eq(1).sum(1)
        interior = (remaining > 0) & (remaining < masked)
        result = fixed_composition_boundary_quota(
            species, logits, target, torch.tensor([site]), generator=generator
        )
        choice = result.species[0, site]
        if interior:
            expected += logits[0, site].log_softmax(-1)[choice].double()
        else:
            forced_logq.append(float(result.log_probability))
        actual += result.log_probability
        species = result.species
    assert actual == pytest.approx(expected)
    assert forced_logq and forced_logq == [0.0] * len(forced_logq)
    assert species.eq(1).sum() == 1


def test_boundary_quota_seed_reproducibility_and_random_orders():
    def sample(seed):
        generator = torch.Generator().manual_seed(seed)
        species = torch.full((2, 8), 2, dtype=torch.long)
        logits = torch.randn(2, 8, 2, generator=generator)
        order = sequential_random_order(2, 8, species.device, generator=generator)
        logq = torch.zeros(2, dtype=torch.float64)
        for step in range(8):
            result = fixed_composition_boundary_quota(
                species, logits, torch.tensor([2, 6]), order[:, step], generator=generator
            )
            species, logq = result.species, logq + result.log_probability
        return species, order, logq

    first, second = sample(81), sample(81)
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))


def test_boundary_quota_invalid_states_raise_immediately():
    species = torch.tensor([[1, 2, 2]])
    logits = torch.zeros(1, 3, 2)
    with pytest.raises(ValueError, match="invalid remaining"):
        fixed_composition_boundary_quota(species, logits, torch.tensor([0]), torch.tensor([1]))
    with pytest.raises(ValueError, match="still be masked"):
        fixed_composition_boundary_quota(species, logits, torch.tensor([1]), torch.tensor([0]))


def test_tau_leap_final_step_clears_all_masks_and_registry_is_complete():
    species = torch.full((3, 108), 2, dtype=torch.long)
    output, logq = janus_tau_leap(species, torch.zeros(3, 108, 2), 0.99, 1.0)
    assert output.ne(2).all()
    assert torch.isfinite(logq).all()
    assert set(DISCRETE_SAMPLER_REGISTRY) == {
        "janus_tau_leap",
        "sequential_random_order",
        "fixed_composition_boundary_quota",
        "fixed_composition_dp",
    }
