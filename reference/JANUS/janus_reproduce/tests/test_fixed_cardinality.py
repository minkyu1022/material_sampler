import itertools

import pytest
import torch

from janus_reproduce.fixed_cardinality import (
    constrained_marginals,
    constrained_nll,
    constrained_reveal_step,
    log_partition,
)


def _enumerate(logits, quota):
    subsets = list(itertools.combinations(range(len(logits)), quota))
    scores = torch.stack([logits[list(subset)].sum() for subset in subsets])
    probabilities = scores.softmax(0)
    marginals = torch.zeros_like(logits)
    for subset, probability in zip(subsets, probabilities, strict=True):
        marginals[list(subset)] += probability
    return subsets, scores.logsumexp(0), probabilities, marginals


@pytest.mark.parametrize("size,quota", [(3, 0), (3, 1), (6, 3), (8, 7), (8, 8)])
def test_dp_partition_marginals_nll_and_gradients_match_exhaustive(size, quota):
    logits = torch.linspace(-1.3, 1.1, size, dtype=torch.float64, requires_grad=True)
    subsets, expected_z, _, expected_marginals = _enumerate(logits, quota)
    actual_z = log_partition(logits, quota)
    actual_marginals = constrained_marginals(logits, quota)
    assert actual_z.item() == pytest.approx(expected_z.item(), abs=1e-12)
    torch.testing.assert_close(actual_marginals, expected_marginals, atol=1e-12, rtol=0)
    torch.testing.assert_close(actual_marginals.sum(), torch.tensor(float(quota), dtype=torch.float64))
    gradient = (
        torch.autograd.grad(actual_z, logits)[0]
        if actual_z.requires_grad
        else torch.zeros_like(logits)
    )
    torch.testing.assert_close(gradient, expected_marginals, atol=1e-12, rtol=0)

    terminal = torch.zeros(1, size, dtype=torch.long)
    terminal[0, list(subsets[-1])] = 1
    binary_logits = torch.stack((torch.zeros_like(logits), logits), -1)[None]
    nll = constrained_nll(binary_logits, terminal, torch.ones_like(terminal, dtype=torch.bool), torch.tensor([quota]))
    expected_nll = expected_z - logits[list(subsets[-1])].sum()
    assert nll.item() == pytest.approx(expected_nll.item(), abs=1e-12)


def test_dp_gradient_matches_finite_difference():
    logits = torch.tensor([-0.9, -0.1, 0.4, 1.2], dtype=torch.float64, requires_grad=True)
    gradient = torch.autograd.grad(log_partition(logits, 2), logits)[0]
    epsilon = 1e-6
    finite_difference = torch.stack(
        [
            (
                log_partition(logits.detach() + torch.eye(4, dtype=torch.float64)[i] * epsilon, 2)
                - log_partition(logits.detach() - torch.eye(4, dtype=torch.float64)[i] * epsilon, 2)
            )
            / (2 * epsilon)
            for i in range(4)
        ]
    )
    torch.testing.assert_close(gradient, finite_difference, atol=1e-9, rtol=1e-7)


def test_equal_logits_have_symmetric_exact_quota_marginals():
    marginals = constrained_marginals(torch.zeros(7), 3)
    torch.testing.assert_close(marginals, torch.full((7,), 3 / 7, dtype=torch.float64))


@pytest.mark.parametrize("quota", [0, 1, 3, 4, 7, 8])
def test_full_reveal_probability_matches_exact_subset_probability(quota):
    size = 8
    logits = torch.linspace(-0.8, 0.9, size)
    binary_logits = torch.stack((torch.zeros_like(logits), logits), -1)[None]
    species, log_probability = constrained_reveal_step(
        torch.full((1, size), 2),
        binary_logits,
        torch.tensor([quota]),
        1.0,
        generator=torch.Generator().manual_seed(40 + quota),
    )
    selected = species[0].eq(1)
    expected = logits[selected].double().sum() - log_partition(logits, quota)
    assert species.eq(1).sum() == quota
    assert log_probability.item() == pytest.approx(expected.item(), abs=2e-7)


def test_sample_frequencies_match_exhaustive_distribution():
    logits = torch.tensor([-0.7, 0.2, 1.1], dtype=torch.float64)
    subsets, _, expected, _ = _enumerate(logits, 2)
    counts = {subset: 0 for subset in subsets}
    generator = torch.Generator().manual_seed(99)
    binary = torch.stack((torch.zeros_like(logits), logits), -1)[None]
    for _ in range(4_000):
        species, _ = constrained_reveal_step(
            torch.full((1, 3), 2), binary, torch.tensor([2]), 1.0, generator=generator
        )
        counts[tuple(torch.where(species[0].eq(1))[0].tolist())] += 1
    observed = torch.tensor([counts[subset] / 4_000 for subset in subsets], dtype=torch.float64)
    torch.testing.assert_close(observed, expected, atol=0.025, rtol=0)


def test_invalid_remaining_quota_fails_loudly():
    with pytest.raises(ValueError, match="invalid remaining"):
        constrained_reveal_step(
            torch.tensor([[1, 1, 2]]), torch.zeros(1, 3, 2), torch.tensor([1]), 1.0
        )


@pytest.mark.parametrize("quota", [0, 1, 32, 64, 96, 127, 128])
def test_n128_terminal_counts_are_exact(quota):
    logits = torch.randn(6, 128, 2, generator=torch.Generator().manual_seed(100 + quota))
    species, log_probability = constrained_reveal_step(
        torch.full((6, 128), 2),
        logits,
        torch.full((6,), quota),
        1.0,
        generator=torch.Generator().manual_seed(200 + quota),
    )
    assert species.eq(1).sum(1).tolist() == [quota] * 6
    assert torch.isfinite(log_probability).all()


def test_multistep_dynamic_logits_match_conditional_subset_probability():
    generator = torch.Generator().manual_seed(314)
    species = torch.full((1, 7), 2)
    target = torch.tensor([3])
    for step in range(12):
        if not species.eq(2).any():
            break
        logits = torch.randn(1, 7, 2, generator=torch.Generator().manual_seed(900 + step))
        before = species.clone()
        species, actual = constrained_reveal_step(
            species, logits, target, 0.35, generator=generator
        )
        newly_revealed = before.eq(2) & species.ne(2)
        masked_before = before[0].eq(2)
        remaining = before[0].eq(2) & ~newly_revealed[0]
        quota_before = int(target[0] - before[0].eq(1).sum())
        selected_cr = species[0, newly_revealed[0]].eq(1)
        remaining_quota = quota_before - int(selected_cr.sum())
        cr_logits = (logits[0, :, 1] - logits[0, :, 0]).double()
        expected = cr_logits[newly_revealed[0]][selected_cr].sum()
        expected += log_partition(cr_logits[remaining], remaining_quota)
        expected -= log_partition(cr_logits[masked_before], quota_before)
        assert actual.item() == pytest.approx(expected.item(), abs=3e-7)
    assert species.eq(1).sum().item() <= 3
