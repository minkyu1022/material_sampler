"""Small torch-only equivariant network for the JANUS alloy channels."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor, nn


def minimum_image_displacements(positions: Tensor, volume: Tensor | float) -> Tensor:
    """Dense pair displacements ``r_i-r_j`` in a periodic cubic cell."""
    length = torch.as_tensor(volume, dtype=positions.dtype, device=positions.device).pow(1 / 3)
    while length.ndim < positions.ndim:
        length = length.unsqueeze(-1)
    displacement = positions.unsqueeze(-2) - positions.unsqueeze(-3)
    length = length.unsqueeze(-2)
    return displacement - length * torch.round(displacement / length)


def minimum_image_displacements_cell(fractional: Tensor, cell: Tensor) -> Tensor:
    """Dense row-vector pair displacements in an orthorhombic periodic cell."""
    displacement = fractional.unsqueeze(-2) - fractional.unsqueeze(-3)
    displacement = displacement - torch.round(displacement)
    return torch.einsum("bijk,bkl->bijl", displacement, cell)


class GaussianRBF(nn.Module):
    def __init__(self, count: int = 16, cutoff: float = 5.0):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, count)
        self.register_buffer("centers", centers)
        self.width = float((count - 1) ** 2 / cutoff**2) if count > 1 else 1.0
        self.cutoff = cutoff

    def forward(self, distance: Tensor) -> tuple[Tensor, Tensor]:
        r = distance.unsqueeze(-1)
        radial = torch.exp(-self.width * (r - self.centers) ** 2)
        envelope = (1 - (distance / self.cutoff).square()).clamp_min(0).pow(3)
        return radial, envelope


class _Interaction(nn.Module):
    def __init__(self, features: int, radial_basis: int):
        super().__init__()
        self.edge = nn.Sequential(
            nn.Linear(radial_basis, features), nn.SiLU(), nn.Linear(features, 3 * features)
        )
        self.message = nn.Linear(features, 3 * features)
        self.vector_mix = nn.Linear(features, 2 * features, bias=False)
        self.update = nn.Sequential(
            nn.Linear(2 * features, features), nn.SiLU(), nn.Linear(features, 3 * features)
        )
        self.scalar_norm = nn.LayerNorm(features)

    def forward(
        self, scalar: Tensor, vector: Tensor, radial: Tensor, envelope: Tensor, unit: Tensor
    ) -> tuple[Tensor, Tensor]:
        edge = self.edge(radial) * envelope[..., None]
        message = self.message(scalar).unsqueeze(-3) * edge
        scalar_message, vector_gate, direction_gate = message.chunk(3, -1)
        scalar = scalar + scalar_message.sum(-2)
        vector = vector + (
            vector_gate.unsqueeze(-2) * vector.unsqueeze(-4)
            + direction_gate.unsqueeze(-2) * unit.unsqueeze(-1)
        ).sum(-3)

        mixed, gate_vector = self.vector_mix(vector).chunk(2, -1)
        invariant = (mixed * gate_vector).sum(-2)
        ds, gate, _ = self.update(torch.cat((scalar, invariant), -1)).chunk(3, -1)
        return self.scalar_norm(scalar + ds), vector + gate.unsqueeze(-2) * mixed


class _VectorHead(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(features, features), nn.SiLU(), nn.Linear(features, 1))
        self.vector = nn.Linear(features, 1, bias=False)
        self.anchor = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.vector.weight)

    def forward(self, scalar: Tensor, vector: Tensor, displacement: Tensor) -> Tensor:
        return (self.vector(vector) * self.gate(scalar).unsqueeze(-2)).squeeze(-1) + (
            self.anchor * displacement
        )


def _zero_head(in_features: int, out_features: int) -> nn.Sequential:
    head = nn.Sequential(
        nn.Linear(in_features, in_features), nn.SiLU(), nn.Linear(in_features, out_features)
    )
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


class AlloyOutput(NamedTuple):
    b_u: Tensor
    s_u: Tensor
    b_v: Tensor
    s_v: Tensor
    species_logits: Tensor


class JANUSAlloy(nn.Module):
    """PaiNN-like shared trunk with the five JANUS alloy outputs.

    ``reference`` and ``displacement`` are fractional cubic-cell coordinates;
    ``log_volume`` is the live ``log(V)`` channel used to rebuild the graph.
    Species ``num_species`` is reserved as the absorbing mask token.
    """

    def __init__(
        self,
        num_species: int = 2,
        *,
        features: int = 64,
        layers: int = 4,
        radial_basis: int = 16,
        cutoff: float = 5.0,
        temperature_reference: float = 900.0,
        temperature_min: float = 600.0,
        temperature_max: float = 1200.0,
        condition_intercept: float = 0.893,
        condition_slope: float = -5.4e-5,
        condition_scale: float = 0.30,
    ):
        super().__init__()
        self.num_species = num_species
        self.temperature_reference = temperature_reference
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max
        self.condition_intercept = condition_intercept
        self.condition_slope = condition_slope
        self.condition_scale = condition_scale
        self.species_embedding = nn.Embedding(num_species + 1, features)
        self.local_features = nn.Linear(num_species + 2, features, bias=False)
        self.condition = nn.Sequential(
            nn.Linear(8, features), nn.SiLU(), nn.Linear(features, features)
        )
        self.rbf = GaussianRBF(radial_basis, cutoff)
        self.interactions = nn.ModuleList(
            _Interaction(features, radial_basis) for _ in range(layers)
        )
        self.b_u = _VectorHead(features)
        self.s_u = _VectorHead(features)
        self.b_v = _zero_head(features, 1)
        self.s_v = _zero_head(features, 1)
        self.species_head = _zero_head(features, num_species)
        self.species_norm = nn.LayerNorm(features)

    @staticmethod
    def _batch(value: Tensor | float, batch: int, like: Tensor) -> Tensor:
        value = torch.as_tensor(value, dtype=like.dtype, device=like.device)
        if value.ndim == 0:
            value = value.expand(batch)
        return value.reshape(batch)

    def forward(
        self,
        species: Tensor,
        displacement: Tensor,
        log_volume: Tensor | float,
        reference: Tensor,
        time: Tensor | float,
        temperature: Tensor | float,
        delta_mu: Tensor | float = 0.0,
    ) -> AlloyOutput:
        unbatched = displacement.ndim == 2
        if unbatched:
            species, displacement = species[None], displacement[None]
        batch, atoms, _ = displacement.shape
        if reference.ndim == 2:
            reference = reference[None].expand(batch, -1, -1)
        log_volume = self._batch(log_volume, batch, displacement)
        time = self._batch(time, batch, displacement)
        temperature = self._batch(temperature, batch, displacement)
        delta_mu = self._batch(delta_mu, batch, displacement)
        if species.shape != (batch, atoms) or reference.shape != displacement.shape:
            raise ValueError("species, displacement and reference shapes do not agree")
        if temperature.le(0).any():
            raise ValueError("temperature must be positive")

        length = log_volume.exp().pow(1 / 3)
        cell = torch.diag_embed(length[:, None].expand(-1, 3))
        condition = self._condition_features(
            time, temperature, delta_mu, log_volume, displacement, species
        )
        scalar, vector = self._encode_state(species, displacement, reference, cell, condition)
        return self._outputs(scalar, vector, displacement, unbatched)

    def _condition_features(
        self,
        time: Tensor,
        temperature: Tensor,
        delta_mu: Tensor,
        log_volume: Tensor,
        displacement: Tensor,
        species: Tensor,
    ) -> Tensor:
        atoms = species.shape[1]
        length = log_volume.exp().pow(1 / 3)
        physical_displacement = displacement * length[:, None, None]
        phase = 2 * math.pi * time
        return torch.stack(
            (
                time,
                phase.sin(),
                phase.cos(),
                (temperature.reciprocal() - 1 / self.temperature_reference)
                / (1 / self.temperature_min - 1 / self.temperature_max),
                (delta_mu - (self.condition_intercept + self.condition_slope * temperature))
                / self.condition_scale,
                ((log_volume - math.log(atoms)) - math.log(11.5)) / 0.1,
                physical_displacement.norm(dim=-1).mean(-1),
                species.eq(self.num_species).float().mean(-1),
            ),
            -1,
        )

    def _encode_state(
        self,
        species: Tensor,
        displacement: Tensor,
        reference: Tensor,
        cell: Tensor,
        condition: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, atoms, _ = displacement.shape
        physical_displacement = torch.einsum("bni,bij->bnj", displacement, cell)
        fractional = reference + displacement
        pair = minimum_image_displacements_cell(fractional, cell)
        distance = pair.square().sum(-1).sqrt()
        unit = pair / distance.clamp_min(1e-8).unsqueeze(-1)
        radial, envelope = self.rbf(distance)
        not_self = ~torch.eye(atoms, dtype=torch.bool, device=species.device)
        envelope = envelope * not_self
        neighbour_species = species[:, None, :]
        coordination = torch.stack(
            tuple(
                (envelope * neighbour_species.eq(kind)).sum(-1)
                for kind in range(self.num_species + 1)
            ),
            -1,
        )
        local_features = torch.cat(
            (physical_displacement.norm(dim=-1, keepdim=True), coordination), -1
        )

        scalar = (
            self.species_embedding(species.long())
            + self.local_features(local_features)
            + self.condition(condition)[:, None]
        )
        vector = torch.zeros(
            batch, atoms, 3, scalar.shape[-1], device=scalar.device, dtype=scalar.dtype
        )
        for interaction in self.interactions:
            scalar, vector = interaction(scalar, vector, radial, envelope, unit)
        return scalar, vector

    def _outputs(
        self, scalar: Tensor, vector: Tensor, displacement: Tensor, unbatched: bool
    ) -> AlloyOutput:
        pooled = scalar.mean(1)
        output = AlloyOutput(
            self.b_u(scalar, vector, displacement),
            self.s_u(scalar, vector, displacement),
            self.b_v(pooled).squeeze(-1),
            self.s_v(pooled).squeeze(-1),
            self.species_head(self.species_norm(scalar.detach())),
        )
        output = AlloyOutput(
            output.b_u - output.b_u.mean(-2, keepdim=True),
            output.s_u - output.s_u.mean(-2, keepdim=True),
            output.b_v,
            output.s_v,
            output.species_logits,
        )
        if unbatched:
            return AlloyOutput(*(item.squeeze(0) for item in output))
        return output


AlloyPaiNN = JANUSAlloy
