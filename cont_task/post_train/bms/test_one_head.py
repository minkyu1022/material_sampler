import pytest
import torch

from one_head import endpoint_to_velocity_score, project_masked_com


def test_endpoint_head_recovers_linear_bridge_targets():
    x0 = torch.tensor([[[1.0, -2.0], [0.5, 3.0]]])
    x1 = torch.tensor([[[4.0, 1.0], [-1.0, 2.0]]])
    t = torch.tensor([0.25])
    xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
    velocity, score = endpoint_to_velocity_score(xt, x1, t, prior_variance=2.0)
    assert torch.allclose(velocity, x1 - x0)
    assert torch.allclose(score, -x0 / ((1 - t[:, None, None]) * 2.0))


def test_mean_mask_and_validation():
    xt = torch.tensor([[2.0]])
    x1 = torch.tensor([[4.0]])
    velocity, score = endpoint_to_velocity_score(
        xt, x1, torch.tensor([0.5]), prior_variance=4.0, prior_mean=1.0
    )
    assert velocity.item() == pytest.approx(4.0)
    assert score.item() == pytest.approx(0.5)
    field = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [9.0, 9.0]]])
    projected = project_masked_com(field, torch.tensor([[False, False, True]]))
    assert torch.allclose(projected[:, :2].sum(1), torch.zeros(1, 2))
    assert torch.equal(projected[:, 2], torch.zeros(1, 2))
    with pytest.raises(ValueError):
        endpoint_to_velocity_score(xt, x1, torch.tensor([0.5]), prior_variance=0.0)
    with pytest.raises(ValueError):
        endpoint_to_velocity_score(xt, x1, torch.tensor([1.0]), prior_variance=1.0)


def test_per_batch_prior_variance_broadcasts_over_sites_and_channels():
    x0 = torch.ones(2, 3, 2)
    x1 = torch.zeros_like(x0)
    t = torch.tensor([0.25, 0.5])
    xt = (1 - t[:, None, None]) * x0
    _, score = endpoint_to_velocity_score(
        xt, x1, t, prior_variance=torch.tensor([2.0, 4.0])
    )
    expected = -x0 / ((1 - t[:, None, None]) * torch.tensor([2.0, 4.0])[:, None, None])
    assert torch.allclose(score, expected)
