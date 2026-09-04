"""Small alloy observables with periodic-cell support."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _array(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def composition(species, elements: Sequence | None = None) -> dict:
    """Return the fraction of atoms of each species."""
    species = _array(species).reshape(-1)
    elements = np.unique(species) if elements is None else elements
    return {element: float(np.mean(species == element)) for element in elements}


def _cell_matrix(cell) -> np.ndarray:
    cell = _array(cell).astype(float)
    if cell.shape == (3,):
        cell = np.diag(cell)
    if cell.shape != (3, 3) or np.linalg.det(cell) == 0:
        raise ValueError("cell must be an invertible length-3 vector or 3x3 matrix")
    return cell


def _minimum_image(displacements, cell) -> np.ndarray:
    cell = _cell_matrix(cell)
    fractional = _array(displacements) @ np.linalg.inv(cell)
    return (fractional - np.round(fractional)) @ cell


def mean_displacement(positions, reference, cell=None) -> float:
    """Mean per-atom displacement, optionally using the periodic minimum image."""
    displacement = _array(positions) - _array(reference)
    if displacement.shape[-1] != 3:
        raise ValueError("positions must end in Cartesian xyz coordinates")
    if cell is not None:
        displacement = _minimum_image(displacement, cell)
    return float(np.linalg.norm(displacement, axis=-1).mean())


def atomic_volume(cell, n_atoms: int) -> float:
    """Cell volume per atom."""
    if n_atoms < 1:
        raise ValueError("n_atoms must be positive")
    return float(abs(np.linalg.det(_cell_matrix(cell))) / n_atoms)


def partial_rdf(positions, species, cell, r_max: float, bins: int = 100):
    """Return ``(r, g)`` where ``g[(a, b)]`` is a periodic partial RDF.

    Unique unordered pairs are used, and each curve is normalized to one for an
    ideal mixture with the observed species counts.
    """
    positions = _array(positions).astype(float)
    species = _array(species).reshape(-1)
    cell = _cell_matrix(cell)
    if positions.shape != (species.size, 3) or r_max <= 0 or bins < 1:
        raise ValueError("positions/species, r_max, or bins are invalid")
    i, j = np.triu_indices(species.size, 1)
    distances = np.linalg.norm(_minimum_image(positions[j] - positions[i], cell), axis=1)
    edges = np.linspace(0.0, r_max, bins + 1)
    shell_volume = 4.0 * np.pi / 3.0 * (edges[1:] ** 3 - edges[:-1] ** 3)
    volume = abs(np.linalg.det(cell))
    curves = {}
    elements = np.unique(species)
    for ai, a in enumerate(elements):
        for b in elements[ai:]:
            mask = ((species[i] == a) & (species[j] == b)) | ((species[i] == b) & (species[j] == a))
            count_a, count_b = np.sum(species == a), np.sum(species == b)
            possible = count_a * (count_a - 1) / 2 if a == b else count_a * count_b
            histogram = np.histogram(distances[mask], edges)[0]
            curves[(a, b)] = (
                histogram * volume / (possible * shell_volume) if possible else np.zeros(bins)
            )
    return (edges[1:] + edges[:-1]) / 2.0, curves


def warren_cowley_sro(positions, species, cell, cutoff: float):
    """Return ``(elements, alpha)`` with ``alpha_ij = 1-P(j|i)/c_j``."""
    positions = _array(positions).astype(float)
    species = _array(species).reshape(-1)
    if positions.shape != (species.size, 3) or cutoff <= 0:
        raise ValueError("positions/species or cutoff are invalid")
    distance = np.linalg.norm(
        _minimum_image(positions[:, None] - positions[None, :], cell), axis=-1
    )
    neighbours = (distance > 0) & (distance < cutoff)
    elements = np.unique(species)
    concentrations = np.array([np.mean(species == element) for element in elements])
    alpha = np.full((elements.size, elements.size), np.nan)
    for row, center in enumerate(elements):
        bonds = neighbours[species == center]
        total = bonds.sum()
        if total:
            probabilities = np.array(
                [bonds[:, species == other].sum() / total for other in elements]
            )
            alpha[row] = 1.0 - probabilities / concentrations
    return elements, alpha


warren_cowley = warren_cowley_sro
