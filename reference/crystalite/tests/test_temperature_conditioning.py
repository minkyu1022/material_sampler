import pytest
import torch

from src.models.embeddings import TemperatureEmbedder


def test_zero_kelvin_and_null_are_distinct_conditions():
    torch.manual_seed(3)
    embed = TemperatureEmbedder(16, reference_k=750.0)
    temperature = torch.tensor([0.0, 0.0, 750.0, 1200.0])
    present = torch.tensor([True, False, False, False])
    result = embed(temperature, present)
    assert not torch.allclose(result[0], result[1])
    assert torch.allclose(result[1], result[2])
    assert torch.allclose(result[2], result[3])


def test_known_temperature_changes_embedding_and_validates_shape():
    torch.manual_seed(4)
    embed = TemperatureEmbedder(8)
    result = embed(torch.tensor([0.0, 750.0]), torch.tensor([True, True]))
    assert not torch.allclose(result[0], result[1])
    with pytest.raises(ValueError):
        embed(torch.zeros(2, 1), torch.ones(2, dtype=torch.bool))
