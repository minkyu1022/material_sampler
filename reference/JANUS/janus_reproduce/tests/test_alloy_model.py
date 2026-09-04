import torch

from janus_reproduce.alloy_model import JANUSAlloy, minimum_image_displacements


def _state(batch=2):
    reference = torch.tensor(
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.0], [0.25, 0.0, 0.25], [0.0, 0.25, 0.25]]
    )
    return (
        torch.tensor([[0, 1, 2, 0]]).expand(batch, -1),
        torch.randn(batch, 4, 3) * 0.01,
        torch.full((batch,), 4.0),
        reference,
    )


def test_dense_minimum_image_uses_live_volume():
    positions = torch.tensor([[[0.1, 0.0, 0.0], [1.9, 0.0, 0.0]]])
    pair = minimum_image_displacements(positions, torch.tensor([8.0]))
    assert pair.shape == (1, 2, 2, 3)
    assert torch.allclose(pair[0, 0, 1], torch.tensor([0.2, 0.0, 0.0]))


def test_alloy_output_shapes_and_zero_initialization():
    model = JANUSAlloy(features=16, layers=2, radial_basis=8)
    species, displacement, log_volume, reference = _state()
    output = model(species, displacement, log_volume, reference, 0.4, 800.0, 0.1)
    assert output.b_u.shape == output.s_u.shape == (2, 4, 3)
    assert output.b_v.shape == output.s_v.shape == (2,)
    assert output.species_logits.shape == (2, 4, 2)
    assert all(torch.count_nonzero(value) == 0 for value in output)


def test_periodic_translation_invariance_and_rotation_equivariance():
    torch.manual_seed(2)
    model = JANUSAlloy(features=12, layers=2, radial_basis=6)
    # Make vector readouts nonzero without changing their equivariant construction.
    torch.nn.init.normal_(model.b_u.vector.weight)
    torch.nn.init.normal_(model.b_u.gate[-1].weight)
    species, displacement, log_volume, reference = _state(batch=1)
    args = (species, displacement, log_volume, reference, 0.3, 700.0, -0.2)
    base = model(*args)
    assert torch.allclose(base.b_u.mean(1), torch.zeros(1, 3), atol=1e-6)
    assert torch.allclose(base.s_u.mean(1), torch.zeros(1, 3), atol=1e-6)
    translated = model(species, displacement, log_volume, reference + 1, 0.3, 700.0, -0.2)
    assert torch.allclose(base.b_u, translated.b_u, atol=2e-5)
    assert torch.allclose(base.species_logits, translated.species_logits, atol=2e-5)

    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated = model(
        species,
        displacement @ rotation.T,
        log_volume,
        reference @ rotation.T,
        0.3,
        700.0,
        -0.2,
    )
    assert torch.allclose(rotated.b_u, base.b_u @ rotation.T, atol=2e-5)
    assert torch.allclose(rotated.species_logits, base.species_logits, atol=2e-5)
