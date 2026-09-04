import json

import pytest

torch = pytest.importorskip("torch")

from janus_reproduce.ising import observables
from janus_reproduce.ising_experiment import (
    IsingExperimentConfig,
    _chain_observables,
    _condition_batch,
    _configuration_examples,
    _damped_train_step,
    _derived_seed,
    _temperature_grid,
    evaluate_checkpoint,
    evaluation_metrics,
    run_experiment,
    sample_in_blocks,
    zero_field_errors,
)


class _AlwaysPlus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, state, time, temperature, delta_mu=0.0):
        return torch.full_like(state, 100.0) + self.anchor * 0


class _ConstantLogits(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(float(value)))

    def forward(self, state, time, temperature, delta_mu=0.0):
        return torch.zeros_like(state) + self.value


def test_zero_field_sampling_restores_exact_spin_symmetry():
    samples = sample_in_blocks(
        _AlwaysPlus(),
        2000,
        4,
        8,
        2.35,
        0.0,
        device=torch.device("cpu"),
        restore_zero_field_symmetry=True,
    )
    assert observables(samples)["up_fraction"] == pytest.approx(0.5, abs=0.03)


def test_training_conditions_cover_coexistence_and_inverse_temperature():
    config = IsingExperimentConfig(batch_size=4000, coexistence_fraction=0.5)
    torch.manual_seed(2)
    temperature, field = _condition_batch(config, torch.device("cpu"))
    assert field.eq(0).float().mean() == pytest.approx(0.5, abs=0.03)
    assert temperature.min() >= config.temperature_min
    assert temperature.max() <= config.temperature_max
    midpoint = 2 / (1 / config.temperature_min + 1 / config.temperature_max)
    assert temperature.median() == pytest.approx(midpoint, abs=0.04)
    assert config.critical_temperature in _temperature_grid(config)


def test_zero_field_acceptance_metrics_focus_on_critical_curve():
    row = {
        "temperature": 2.35,
        "delta_mu": 0.0,
        "janus": {"up_fraction": 0.49, "abs_magnetization": 0.60},
        "wolff": {"up_fraction": 0.50, "abs_magnetization": 0.59},
    }
    metrics = zero_field_errors([row])
    assert metrics["critical_abs_magnetization_mae"] == pytest.approx(0.01)
    assert metrics["zero_field_spin_symmetry_error"] == pytest.approx(0.01)


def test_grid_seeds_are_reproducible_and_condition_specific():
    assert _derived_seed(7, 2, 3, 1) == _derived_seed(7, 2, 3, 1)
    assert len({_derived_seed(7, 2, 3, stream) for stream in range(2)}) == 2
    assert _derived_seed(7, 2, 3, 1) != _derived_seed(7, 2, 4, 1)


def test_chain_observables_and_grid_metrics_preserve_uncertainty():
    reference = torch.tensor(
        [
            [[[-1, -1], [-1, -1]], [[1, 1], [1, 1]]],
            [[[-1, -1], [-1, -1]], [[1, 1], [1, 1]]],
        ]
    ).numpy()
    by_chain, standard_error = _chain_observables(reference, 0.0)
    assert len(by_chain) == 2
    assert standard_error["magnetization"] == pytest.approx(1.0)
    row = {
        "temperature": 2.2692,
        "delta_mu": 0.0,
        "janus": {key: value + 0.1 for key, value in observables(reference).items()},
        "wolff": observables(reference),
        "wolff_chain_standard_error": standard_error,
    }
    metrics = evaluation_metrics([row])
    assert metrics["max_grid_up_fraction_error"] == pytest.approx(0.1)
    assert metrics["zero_field_by_temperature"][0]["temperature"] == 2.2692
    assert metrics["wolff_chain_uncertainty"]["magnetization"]["max_standard_error"] == 1


def test_configuration_examples_cover_below_near_and_above_tc():
    config = IsingExperimentConfig(length=4, reveal_steps=4)
    examples = _configuration_examples(_AlwaysPlus(), config, torch.device("cpu"))
    assert list(examples) == ["below Tc", "near Tc", "above Tc"]
    assert [temperature for temperature, _ in examples.values()] == [1.5, 2.2692, 3.2]


def test_damping_adds_logit_mse_on_the_same_masked_states():
    terminals = torch.ones(32, 2, 2)
    temperature = torch.full((32,), 2.0)
    delta_mu = torch.zeros(32)
    model = _ConstantLogits(1)
    previous = _ConstantLogits(0)
    previous.requires_grad_(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0)

    torch.manual_seed(3)
    undamped = _damped_train_step(model, previous, optimizer, terminals, temperature, delta_mu, 0)
    torch.manual_seed(3)
    damped = _damped_train_step(model, previous, optimizer, terminals, temperature, delta_mu, 10)
    assert damped - undamped == pytest.approx(10)


def test_damping_config_validation():
    with pytest.raises(ValueError, match="damping_eta"):
        IsingExperimentConfig(damping_eta=-1)
    with pytest.raises(ValueError, match="counts"):
        IsingExperimentConfig(previous_model_interval=0)


def test_smoke_experiment_saves_complete_artifacts(tmp_path):
    config = IsingExperimentConfig(
        length=2,
        temperature_points=2,
        delta_mu_points=2,
        reveal_steps=2,
        rounds=1,
        batch_size=2,
        gradient_steps=1,
        eval_samples=2,
        reference_samples=2,
        reference_burn_in=0,
        reference_chains=1,
        width=4,
        depth=1,
    )
    result = run_experiment(config, tmp_path, device="cpu")
    assert len(result["grid"]) == 6
    assert json.loads((tmp_path / "resolved_config.json").read_text())["length"] == 2
    for name in (
        "checkpoint.pt",
        "metrics.json",
        "population_map.png",
        "abs_magnetization.png",
        "config_examples.png",
    ):
        assert (tmp_path / name).stat().st_size > 0

    checkpoint = torch.load(tmp_path / "checkpoint.pt", weights_only=False)
    assert checkpoint["previous_model"].keys() == checkpoint["model"].keys()
    assert result["provenance"]["reference_protocol"]["cluster_steps_per_chain_total"] == 2
    assert result["grid"][0]["wolff_chain_observables"]

    resumed = run_experiment(config, tmp_path, resume=True, device="cpu")
    assert len(resumed["grid"]) == 6


def test_checkpoint_evaluation_does_not_mutate_training_checkpoint(tmp_path):
    training = tmp_path / "training"
    evaluation = tmp_path / "evaluation"
    config = IsingExperimentConfig(
        length=2,
        temperature_points=2,
        delta_mu_points=2,
        reveal_steps=2,
        rounds=1,
        batch_size=2,
        gradient_steps=1,
        eval_samples=2,
        reference_samples=2,
        reference_burn_in=0,
        reference_chains=1,
        width=4,
        depth=1,
    )
    run_experiment(config, training, device="cpu")
    checkpoint = training / "checkpoint.pt"
    before = checkpoint.read_bytes()
    result = evaluate_checkpoint(checkpoint, config, evaluation, device="cpu")
    assert checkpoint.read_bytes() == before
    assert result["provenance"]["checkpoint"] == str(checkpoint.resolve())
    assert (evaluation / "resolved_eval_config.json").is_file()
    assert not (evaluation / "checkpoint.pt").exists()
