"""Common N=128 BCT state space for the unified Ni--Cr benchmark."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn

from .alloy_model import JANUSAlloy, _zero_head

N_ATOMS = 128
REPEATS = 4
L_REF = 2.765
GRAPH_CUTOFF = 5.3  # New-model baseline; not a paper-faithful cutoff claim.


@dataclass(frozen=True)
class CellNormalization:
    """Guideline-endpoint calibration: BCC and FCC-equivalent map to opposite unit z."""

    mean: tuple[float, float] = (-0.058033303606018175, 0.11528310318584964)
    scale: tuple[float, float] = (0.058033303606018175, 0.11528310318584964)

    def encode(self, cell_ac: Tensor) -> Tensor:
        mean = cell_ac.new_tensor(self.mean)
        scale = cell_ac.new_tensor(self.scale)
        return (torch.log(cell_ac / L_REF) - mean) / scale

    def decode(self, cell_z: Tensor) -> Tensor:
        mean = cell_z.new_tensor(self.mean)
        scale = cell_z.new_tensor(self.scale)
        return L_REF * torch.exp(mean + scale * cell_z)


@dataclass(frozen=True)
class BCTDomain:
    """Explicit solid-Bain basin that makes the constrained target normalizable."""

    primitive_volume_min: float = 16.0
    primitive_volume_max: float = 28.0
    ratio_min: float = 0.95
    ratio_max: float = 1.48
    # Below half the BCC reference-site nearest-neighbour separation in full-cell
    # fractional coordinates; site anchoring is tracked independently as well.
    rms_u_max: float = 0.09

    def contains(self, cell_ac: Tensor, disp_u: Tensor | None = None) -> Tensor:
        volume = cell_ac[..., 0].square() * cell_ac[..., 1]
        ratio = cell_ac[..., 1] / cell_ac[..., 0]
        valid = (
            (volume >= self.primitive_volume_min)
            & (volume <= self.primitive_volume_max)
            & (ratio >= self.ratio_min)
            & (ratio <= self.ratio_max)
        )
        if disp_u is not None:
            rms = disp_u.square().sum(-1).mean(-1).sqrt()
            valid &= rms <= self.rms_u_max
            reference = reference_sites(dtype=disp_u.dtype).to(disp_u.device)
            delta = reference[:, None] + disp_u[..., :, None, :] - reference[None, :]
            delta -= delta.round()
            distances = torch.einsum(
                "...nmi,...ij->...nmj", delta, cell_matrix(cell_ac)
            ).square().sum(-1)
            expected = torch.arange(N_ATOMS, device=disp_u.device)
            valid &= distances.argmin(-1).eq(expected).all(-1)
        return valid


DEFAULT_BCT_DOMAIN = BCTDomain()


def reference_sites(*, dtype: torch.dtype = torch.float64) -> Tensor:
    """Fixed fractional sites in the full 4x4x4 BCT supercell."""
    grid = torch.cartesian_prod(*(torch.arange(REPEATS),) * 3).to(dtype)
    basis = torch.tensor(((0, 0, 0), (0.5, 0.5, 0.5)), dtype=dtype)
    return ((grid[:, None, :] + basis[None, :, :]) / REPEATS).reshape(N_ATOMS, 3)


def cell_matrix(cell_ac: Tensor) -> Tensor:
    """Full-supercell row-vector matrix from primitive lengths ``(a, c)``."""
    cell_ac = torch.as_tensor(cell_ac)
    if cell_ac.shape[-1] != 2 or torch.any(cell_ac <= 0):
        raise ValueError("cell_ac must end in two positive primitive lengths (a, c)")
    output = torch.zeros(*cell_ac.shape[:-1], 3, 3, dtype=cell_ac.dtype, device=cell_ac.device)
    output[..., 0, 0] = REPEATS * cell_ac[..., 0]
    output[..., 1, 1] = REPEATS * cell_ac[..., 0]
    output[..., 2, 2] = REPEATS * cell_ac[..., 1]
    return output


def log_coordinate_jacobian(cell_ac: Tensor) -> Tensor:
    """Nonconstant ``log|d(a,c)/d(y_a,y_c)|`` contribution."""
    return torch.log(cell_ac).sum(-1)


def transformed_target_log_density(
    energy: Tensor,
    cell_ac: Tensor,
    beta: Tensor | float,
    *,
    pressure: Tensor | float = 0.0,
    n_atoms: int = N_ATOMS,
    disp_u: Tensor | None = None,
    domain: BCTDomain | None = DEFAULT_BCT_DOMAIN,
) -> Tensor:
    """Restricted NPT target in normalized cell coordinates, up to constants.

    Zero-COM fractional coordinates contribute ``V**(N-1)``; transforming
    ``da dc`` to log lengths contributes one additional factor ``a*c``.
    """
    matrix = cell_matrix(cell_ac)
    volume = torch.linalg.det(matrix)
    beta = torch.as_tensor(beta, dtype=energy.dtype, device=energy.device)
    pressure = torch.as_tensor(pressure, dtype=energy.dtype, device=energy.device)
    value = (
        -beta * (energy + pressure * volume)
        # Zero-COM removes one three-dimensional translation mode, so the
        # fractional-to-Cartesian Jacobian is det(L)^(N-1), not det(L)^N.
        + (n_atoms - 1) * torch.log(volume)
        + log_coordinate_jacobian(cell_ac)
    )
    if domain is not None:
        value = torch.where(domain.contains(cell_ac, disp_u), value, -torch.inf)
    return value


class UnifiedBCT2DOutput(NamedTuple):
    b_u: Tensor
    s_u: Tensor
    b_cell: Tensor
    s_cell: Tensor
    species_logits: Tensor


class JANUSUnifiedBCT2D(JANUSAlloy):
    """Cu--Ni-validated PaiNN trunk with a normalized two-length cell channel."""

    def __init__(self, *, normalization: CellNormalization | None = None, **kwargs):
        kwargs.setdefault("cutoff", GRAPH_CUTOFF)
        kwargs.setdefault("temperature_reference", 1050.0)
        kwargs.setdefault("temperature_min", 600.0)
        kwargs.setdefault("temperature_max", 1500.0)
        super().__init__(condition_intercept=0, condition_slope=0, condition_scale=1, **kwargs)
        features = self.species_embedding.embedding_dim
        self.condition = nn.Sequential(nn.Linear(9, features), nn.SiLU(), nn.Linear(features, features))
        self.b_v = _zero_head(features, 2)
        self.s_v = _zero_head(features, 2)
        self.normalization = normalization or CellNormalization()

    def forward(
        self,
        species: Tensor,
        disp_u: Tensor,
        cell_z: Tensor,
        reference: Tensor,
        time: Tensor | float,
        temperature: Tensor | float,
        cr_fraction: Tensor | float,
    ) -> UnifiedBCT2DOutput:
        unbatched = disp_u.ndim == 2
        if unbatched:
            species, disp_u, cell_z = species[None], disp_u[None], cell_z[None]
        batch, atoms, _ = disp_u.shape
        if atoms != N_ATOMS:
            raise ValueError(f"unified production model requires {N_ATOMS} atoms")
        if reference.ndim == 2:
            reference = reference[None].expand(batch, -1, -1)
        time = self._batch(time, batch, disp_u)
        temperature = self._batch(temperature, batch, disp_u)
        cr_fraction = self._batch(cr_fraction, batch, disp_u)
        if species.shape != (batch, atoms) or reference.shape != disp_u.shape or cell_z.shape != (batch, 2):
            raise ValueError("species, displacement, cell_z, and reference shapes do not agree")
        cell_ac = self.normalization.decode(cell_z)
        cell = cell_matrix(cell_ac)
        physical_u = torch.einsum("bni,bij->bnj", disp_u, cell)
        phase = 2 * math.pi * time
        condition = torch.stack(
            (
                time,
                phase.sin(),
                phase.cos(),
                (temperature.reciprocal() - 1 / self.temperature_reference)
                / (1 / self.temperature_min - 1 / self.temperature_max),
                cr_fraction,
                cell_z[:, 0],
                cell_z[:, 1],
                physical_u.norm(dim=-1).mean(-1),
                species.eq(self.num_species).float().mean(-1),
            ),
            -1,
        )
        scalar, vector = self._encode_state(species, disp_u, reference, cell, condition)
        pooled = scalar.mean(1)
        b_u_cartesian = self.b_u(scalar, vector, physical_u)
        s_u_cartesian = self.s_u(scalar, vector, physical_u)
        b_u_fractional = torch.linalg.solve(
            cell.transpose(-1, -2), b_u_cartesian.transpose(-1, -2)
        ).transpose(-1, -2)
        s_u_fractional = torch.einsum("bni,bji->bnj", s_u_cartesian, cell)
        output = UnifiedBCT2DOutput(
            b_u_fractional,
            s_u_fractional,
            self.b_v(pooled),
            self.s_v(pooled),
            self.species_head(self.species_norm(scalar.detach())),
        )
        output = UnifiedBCT2DOutput(
            output.b_u - output.b_u.mean(-2, keepdim=True),
            output.s_u - output.s_u.mean(-2, keepdim=True),
            output.b_cell,
            output.s_cell,
            output.species_logits,
        )
        return UnifiedBCT2DOutput(*(x.squeeze(0) for x in output)) if unbatched else output
