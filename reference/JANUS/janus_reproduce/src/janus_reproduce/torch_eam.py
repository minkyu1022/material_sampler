"""Torch-native evaluator for tabulated Cu--Ni setfl EAM potentials."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from scipy.interpolate import InterpolatedUnivariateSpline, PPoly
from torch import Tensor, nn


class TorchEAMLabels(NamedTuple):
    energy: Tensor
    forces: Tensor
    stress: Tensor
    log_volume_derivative: Tensor


class TorchEAMCellLabels(NamedTuple):
    energy: Tensor
    forces: Tensor
    cell_derivative: Tensor


def _spline_coefficients(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ASE-compatible not-a-knot spline knots and local cubic coefficients."""
    polynomial = PPoly.from_spline(InterpolatedUnivariateSpline(x, y, k=3)._eval_args)
    # PPoly contains repeated end knots inherited from the B-spline representation.
    keep = np.diff(polynomial.x) > 0
    return polynomial.x[:-1][keep], polynomial.c[:, keep].T


def _read_setfl(path: Path) -> dict[str, object]:
    lines = path.read_text().splitlines()
    if len(lines) < 6:
        raise ValueError(f"invalid setfl potential: {path}")
    data = " ".join(lines[3:]).split()
    count = int(data[0])
    elements = tuple(data[1 : 1 + count])
    cursor = 1 + count
    nrho, drho, nr, dr, cutoff = (
        int(data[cursor]),
        float(data[cursor + 1]),
        int(data[cursor + 2]),
        float(data[cursor + 3]),
        float(data[cursor + 4]),
    )
    cursor += 5
    embedding = np.empty((count, nrho))
    finnis_sinclair = path.name.endswith((".eam.fs", ".eam.fs.txt"))
    density = np.empty((count, count, nr))
    for element in range(count):
        cursor += 4  # atomic number, mass, lattice constant, lattice type
        embedding[element] = np.asarray(data[cursor : cursor + nrho], dtype=float)
        cursor += nrho
        if finnis_sinclair:
            for neighbour in range(count):
                density[element, neighbour] = np.asarray(data[cursor : cursor + nr], dtype=float)
                cursor += nr
        else:
            values = np.asarray(data[cursor : cursor + nr], dtype=float)
            cursor += nr
            density[:, element] = values
    rphi = np.zeros((count, count, nr))
    for first in range(count):
        for second in range(first + 1):
            values = np.asarray(data[cursor : cursor + nr], dtype=float)
            cursor += nr
            rphi[second, first] = rphi[first, second] = values
    if cursor != len(data):
        raise ValueError(f"unexpected trailing data in setfl potential: {path}")
    return {
        "elements": elements,
        "nrho": nrho,
        "drho": drho,
        "nr": nr,
        "dr": dr,
        "cutoff": cutoff,
        "embedding": embedding,
        "density": density,
        "rphi": rphi,
    }


class TorchEAM(nn.Module):
    """Differentiable batched setfl/EAM-FS evaluator for periodic cubic cells.

    Species follow the potential-file order (``0=Cu, 1=Ni``). Positions are
    fractional cubic-cell coordinates and ``log_volume`` is log(cell volume).
    Float64 reproduces ASE's setfl evaluator most closely; ``module.to(dtype=...)``
    may be used when throughput matters more than reference-level agreement.
    """

    def __init__(
        self,
        potential: str | Path,
        *,
        elements: tuple[str, ...] | None = None,
        species_indices: tuple[int, ...] | None = None,
        cutoff: float | None = None,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        path = Path(potential).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        raw = _read_setfl(path)
        file_elements = tuple(raw["elements"])
        if elements is not None and file_elements != elements:
            raise ValueError(f"expected {elements} element order, found {file_elements}")
        self.elements = file_elements
        self.n_elements = len(file_elements)
        self.species_indices = species_indices or tuple(range(self.n_elements))
        if not self.species_indices or min(self.species_indices) < 0 or max(self.species_indices) >= self.n_elements:
            raise ValueError("species_indices must select potential-file elements")
        self.n_species = len(self.species_indices)
        self.path = path
        self.native_cutoff = float(raw["cutoff"])
        self.cutoff = self.native_cutoff if cutoff is None else float(cutoff)
        if not 0 < self.cutoff <= self.native_cutoff:
            raise ValueError(f"cutoff must be in (0, {self.native_cutoff}]")
        r = np.arange(int(raw["nr"])) * float(raw["dr"])
        rho = np.arange(int(raw["nrho"])) * float(raw["drho"])

        splines: list[tuple[np.ndarray, np.ndarray]] = []
        for values in raw["embedding"]:
            splines.append(_spline_coefficients(rho, values))
        for values in np.asarray(raw["density"]).reshape(-1, int(raw["nr"])):
            splines.append(_spline_coefficients(r, values))
        rphi = raw["rphi"]
        for first, second in itertools.product(range(self.n_elements), repeat=2):
            splines.append(_spline_coefficients(r[1:], rphi[first, second, 1:] / r[1:]))

        width = max(len(item[0]) for item in splines)
        coefficients = np.zeros((len(splines), width, 4))
        origins = np.empty(len(splines))
        steps = np.empty(len(splines))
        counts = np.empty(len(splines), dtype=int)
        for index, (spline_knots, spline_coefficients) in enumerate(splines):
            coefficients[index, : len(spline_coefficients)] = spline_coefficients
            origins[index] = spline_knots[0]
            steps[index] = (spline_knots[1] - spline_knots[0]) / 2
            counts[index] = len(spline_knots)
            expected = origins[index] + steps[index] * np.r_[0, np.arange(2, counts[index] + 1)]
            if not np.allclose(spline_knots, expected, rtol=1e-10, atol=1e-12):
                raise ValueError("TorchCuNiEAM requires uniformly tabulated setfl splines")
        self.register_buffer("coefficients", torch.as_tensor(coefficients, dtype=dtype))
        self.register_buffer("origins", torch.as_tensor(origins, dtype=dtype))
        self.register_buffer("steps", torch.as_tensor(steps, dtype=dtype))
        self.register_buffer("counts", torch.as_tensor(counts, dtype=torch.long))
        self.register_buffer("species_map", torch.as_tensor(self.species_indices, dtype=torch.long))

    def _spline(self, table: int | Tensor, value: Tensor) -> Tensor:
        table = torch.as_tensor(table, device=value.device, dtype=torch.long)
        while table.ndim < value.ndim:
            table = table.unsqueeze(0)
        table = table.expand(value.shape)
        origin, step, count = self.origins[table], self.steps[table], self.counts[table]
        index = (torch.floor((value - origin) / step).long() - 1).clamp_min(0)
        index = torch.minimum(index, count - 1)
        x0 = torch.where(index.eq(0), origin, origin + (index + 1) * step)
        coefficient = self.coefficients[table, index]
        delta = value - x0
        return (
            (coefficient[..., 0] * delta + coefficient[..., 1]) * delta + coefficient[..., 2]
        ) * delta + coefficient[..., 3]

    def _batch(self, species: Tensor, fractional: Tensor, log_volume: Tensor | float):
        unbatched = fractional.ndim == 2
        if unbatched:
            species, fractional = species[None], fractional[None]
        log_volume = torch.as_tensor(log_volume, dtype=fractional.dtype, device=fractional.device)
        if log_volume.ndim == 0:
            log_volume = log_volume.expand(fractional.shape[0])
        if species.shape != fractional.shape[:2] or log_volume.shape != fractional.shape[:1]:
            raise ValueError("species, fractional positions, and log_volume shapes do not agree")
        if species.lt(0).any() or species.ge(self.n_species).any():
            raise ValueError(f"species must be indices into selected elements {self.species_indices}")
        return self.species_map[species.long()], fractional, log_volume, unbatched

    def _radials(
        self,
        fractional: Tensor,
        log_volume: Tensor,
        strain: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        length = log_volume.exp().pow(1 / 3)
        cell = torch.diag_embed(length[:, None].expand(-1, 3))
        return self._radials_cell(fractional, cell, strain)

    def _radials_cell(
        self,
        fractional: Tensor,
        cell: Tensor,
        strain: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        atoms = fractional.shape[1]
        shortest = torch.linalg.vector_norm(cell, dim=-1).min()
        max_image = int(torch.ceil(self.cutoff / shortest.detach()).item())
        shifts = torch.cartesian_prod(
            *(torch.arange(-max_image, max_image + 1, device=fractional.device),) * 3
        ).to(fractional.dtype)
        relative = fractional[:, None, :, None, :] - fractional[:, :, None, None, :]
        relative = torch.einsum(
            "bijnk,bkl->bijnl", relative + shifts[None, None, None], cell
        )
        if strain is not None:
            deformation = torch.eye(3, dtype=fractional.dtype, device=fractional.device) + strain
            relative = torch.einsum("b...i,bij->b...j", relative, deformation)
        distance = relative.square().sum(-1).clamp_min(torch.finfo(relative.dtype).tiny).sqrt()
        central = shifts.eq(0).all(-1)
        not_self = ~torch.eye(atoms, dtype=torch.bool, device=fractional.device)
        valid = distance.le(self.cutoff) & (~central[None, None, None] | not_self[None, :, :, None])

        count = self.n_elements
        start = count
        radial_tables = torch.arange(start, start + 2 * count**2, device=fractional.device)
        radial = self._spline(
            radial_tables, distance[..., None].expand(*distance.shape, 2 * count**2)
        )
        radial = radial * valid[..., None]
        density = radial[..., : count**2].sum(3)
        pair = radial[..., count**2 :].sum(3)
        return density, pair

    def _batch_cell(self, species: Tensor, fractional: Tensor, cell: Tensor):
        unbatched = fractional.ndim == 2
        if unbatched:
            species, fractional = species[None], fractional[None]
        cell = torch.as_tensor(cell, dtype=fractional.dtype, device=fractional.device)
        if cell.ndim == 2:
            cell = cell[None].expand(fractional.shape[0], -1, -1)
        if species.shape != fractional.shape[:2] or cell.shape != (len(species), 3, 3):
            raise ValueError("species, fractional positions, and cell shapes do not agree")
        if species.lt(0).any() or species.ge(self.n_species).any():
            raise ValueError(f"species must be indices into selected elements {self.species_indices}")
        return self.species_map[species.long()], fractional, cell, unbatched

    def _energies(self, species: Tensor, density: Tensor, pair: Tensor) -> Tensor:
        geometries = density.shape[0]
        configs = species.shape[0]
        if geometries not in (1, configs):
            raise ValueError("geometry batch must be one or match species batch")
        density = density.expand(configs, -1, -1, -1)
        pair = pair.expand(configs, -1, -1, -1)
        pair_type = self.n_elements * species[:, :, None] + species[:, None, :]
        electron_density = density.gather(-1, pair_type.unsqueeze(-1)).squeeze(-1).sum(-1)
        embedding = self._spline(species, electron_density).sum(-1)
        pair_energy = pair.gather(-1, pair_type.unsqueeze(-1)).squeeze(-1).sum((-1, -2)) / 2
        return embedding + pair_energy

    def forward(self, species: Tensor, fractional: Tensor, log_volume: Tensor | float) -> Tensor:
        species, fractional, log_volume, unbatched = self._batch(species, fractional, log_volume)
        energy = self._energies(species, *self._radials(fractional, log_volume))
        return energy.squeeze(0) if unbatched else energy

    def forward_cell(self, species: Tensor, fractional: Tensor, cell: Tensor) -> Tensor:
        """Energy in a general periodic cell using row-vector fractional coordinates."""
        species, fractional, cell, unbatched = self._batch_cell(species, fractional, cell)
        energy = self._energies(species, *self._radials_cell(fractional, cell))
        return energy.squeeze(0) if unbatched else energy

    def all_site_energies(
        self, species: Tensor, fractional: Tensor, log_volume: Tensor | float
    ) -> Tensor:
        """Return total energies with every site independently set to each selected element."""
        species, fractional, log_volume, unbatched = self._batch(species, fractional, log_volume)
        outputs = []
        for batch_index in range(species.shape[0]):
            atoms = species.shape[1]
            alternatives = species[batch_index].expand(self.n_species * atoms, -1).clone()
            sites = torch.arange(atoms, device=species.device).repeat_interleave(self.n_species)
            alternatives[
                torch.arange(self.n_species * atoms, device=species.device), sites
            ] = self.species_map.repeat(atoms)
            radial = self._radials(
                fractional[batch_index : batch_index + 1], log_volume[batch_index : batch_index + 1]
            )
            outputs.append(self._energies(alternatives, *radial).reshape(atoms, self.n_species))
        result = torch.stack(outputs)
        return result.squeeze(0) if unbatched else result

    def labels(
        self,
        species: Tensor,
        fractional: Tensor,
        log_volume: Tensor | float,
        *,
        create_graph: bool = False,
    ) -> TorchEAMLabels:
        """Return energy, Cartesian forces, virial stress, and dE/dlog(V)."""
        species, fractional, log_volume, unbatched = self._batch(species, fractional, log_volume)
        fractional = fractional.requires_grad_(True)
        log_volume = log_volume.requires_grad_(True)
        strain = torch.zeros(
            species.shape[0],
            3,
            3,
            dtype=fractional.dtype,
            device=fractional.device,
            requires_grad=True,
        )
        energy = self._energies(species, *self._radials(fractional, log_volume, strain))
        grad_fractional, grad_volume, grad_strain = torch.autograd.grad(
            energy.sum(), (fractional, log_volume, strain), create_graph=create_graph
        )
        length = log_volume.exp().pow(1 / 3)
        forces = -grad_fractional / length[:, None, None]
        stress = grad_strain / log_volume.exp()[:, None, None]
        labels = TorchEAMLabels(energy, forces, stress, grad_volume)
        if unbatched:
            return TorchEAMLabels(*(value.squeeze(0) for value in labels))
        return labels

    def labels_cell(
        self,
        species: Tensor,
        fractional: Tensor,
        cell: Tensor,
        *,
        create_graph: bool = False,
    ) -> TorchEAMCellLabels:
        """Energy, Cartesian forces, and fixed-fractional-coordinate cell derivative."""
        species, fractional, cell, unbatched = self._batch_cell(species, fractional, cell)
        fractional = fractional.requires_grad_(True)
        cell = cell.requires_grad_(True)
        energy = self._energies(species, *self._radials_cell(fractional, cell))
        grad_fractional, grad_cell = torch.autograd.grad(
            energy.sum(), (fractional, cell), create_graph=create_graph
        )
        forces = -torch.linalg.solve(cell, grad_fractional.transpose(-1, -2)).transpose(-1, -2)
        labels = TorchEAMCellLabels(energy, forces, grad_cell)
        if unbatched:
            return TorchEAMCellLabels(*(value.squeeze(0) for value in labels))
        return labels


class TorchCuNiEAM(TorchEAM):
    """Backward-compatible Cu--Ni specialization."""

    def __init__(self, potential: str | Path, *, dtype: torch.dtype = torch.float64):
        super().__init__(potential, elements=("Cu", "Ni"), dtype=dtype)
