import numpy as np
import pytest
from ase.calculators.calculator import Calculator, all_changes

from janus_reproduce.cuni import (
    KB_EV_K,
    CuNiEAM,
    build_cuni_fcc,
    delta_mu_108,
    fit_displacement_width,
    fit_volume_prior,
    quasiharmonic_width,
    temperatures_108,
)


class ToyCalculator(Calculator):
    implemented_properties = ("energy", "forces", "stress")
    elements = ("Cu", "Ni")

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        cu = np.asarray(atoms.get_chemical_symbols()) == "Cu"
        self.results = {
            "energy": float(np.square(atoms.positions).sum() + 2 * cu.sum()),
            "forces": -2 * atoms.positions,
            "stress": np.zeros(6),
        }


def test_cells_and_published_108_grid():
    assert len(build_cuni_fcc(108, cu_fraction=0.25)) == 108
    assert len(build_cuni_fcc(256, cu_fraction=0.25)) == 256
    assert build_cuni_fcc(108, cu_fraction=0.25).symbols.count("Cu") == 27
    assert np.array_equal(temperatures_108(), np.arange(500, 1250, 50))
    assert len(delta_mu_108()) == 33
    assert delta_mu_108()[[0, -1]] == pytest.approx([0.6, 1.15])


def test_si_prior_fits_recover_synthetic_parameters():
    composition, temperature = np.meshgrid(np.linspace(0, 1, 5), [600, 900, 1200])
    composition, temperature = composition.ravel(), temperature.ravel()
    volume = (10 * (1 - composition) + 12 * composition + 0.8 * composition * (1 - composition)) * (
        1 + 2e-5 * (temperature - 900)
    )
    fit = fit_volume_prior(composition, temperature, volume)
    assert (fit.v_ni, fit.v_cu, fit.omega, fit.alpha) == pytest.approx(
        (10, 12, 0.8, 2e-5), abs=1e-9
    )
    temperatures = np.array([600.0, 900.0, 1200.0])
    sigma_ref, exponent = fit_displacement_width(temperatures, 0.02 * (temperatures / 900) ** 0.5)
    assert (sigma_ref, exponent) == pytest.approx((0.02, 0.5))
    assert quasiharmonic_width(np.eye(6) * 2, 900) == pytest.approx(np.sqrt(KB_EV_K * 900 / 2))


def test_oracle_heat_bath_labels_are_all_site_conditionals():
    oracle = CuNiEAM.__new__(CuNiEAM)
    oracle.calculator = ToyCalculator()
    atoms = build_cuni_fcc(108, cu_fraction=0.5)
    labels = oracle.labels(atoms, temperature=900, delta_mu=2.0)
    assert labels.forces.shape == (108, 3)
    assert labels.stress.shape == (3, 3)
    assert labels.substitution_energies == pytest.approx(2.0)
    assert labels.heat_bath[:, 1] == pytest.approx(0.5)
    shifted = oracle.labels(atoms, temperature=900, delta_mu=2.0 + KB_EV_K * 900)
    assert shifted.heat_bath[:, 1] == pytest.approx(1 / (1 + np.exp(-1)))
