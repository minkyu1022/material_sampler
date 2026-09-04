# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Fully-connected, E(3)-equivariant PaiNN backbone.

The network maps ``(time, positions)`` to a per-atom vector field that the
``ClippedModel`` controller turns into the SDE control. Distances are computed
densely (all pairs), which is efficient for the small molecules studied here.
"""

from contextlib import nullcontext
from typing import Literal

import torch
import torch.nn as nn

from bms.model.layer.embedding_block import AtomEmbedding, TimeEmbedding
from bms.model.layer.radial_basis import RadialBasis
from bms.utils.composition import PeriodicTable, composition_to_atomic_numbers


class ParityBreakingMessageBlock(nn.Module):
    def __init__(self, num_features: int, num_radial_basis: int):
        super().__init__()
        self.num_features = num_features

        self.mlp_phi = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features * 4),
        )
        self.linear_W = nn.Linear(num_radial_basis, num_features * 4)

    def forward(
        self,
        s: torch.Tensor,  # [B, N, 1, n_feats]
        v: torch.Tensor,  # [B, N, 3, n_feats]
        radial_embeddings: torch.Tensor,  # [B, N, N, 1, num_radial_basis]
        envelope: torch.Tensor,  # [B, N, N, 1]
        unit_vectors: torch.Tensor,  # [B, N, N, 3]
    ):
        phi = self.mlp_phi(s)
        W = self.linear_W(radial_embeddings) * envelope[..., None]
        x = phi[:, None, ...] * W
        x_s, x_vv, x_vs, x_vc = torch.split(x, self.num_features, dim=-1)
        ds = torch.sum(x_s, dim=2)  # sum over j
        x_v = (
            v[:, None, ...] * x_vv
            + x_vs * unit_vectors[..., None]
            + x_vc * torch.cross(v[:, None, ...], unit_vectors[..., None], dim=-2)
        )
        dv = torch.sum(x_v, dim=2)  # sum over j
        return s + ds, v + dv


class MessageBlock(nn.Module):
    def __init__(self, num_features: int, num_radial_basis: int):
        super().__init__()
        self.num_features = num_features

        self.mlp_phi = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features * 3),
        )
        self.linear_W = nn.Linear(num_radial_basis, num_features * 3)

    def forward(
        self,
        s: torch.Tensor,
        v: torch.Tensor,
        radial_embeddings: torch.Tensor,
        envelope: torch.Tensor,
        unit_vectors: torch.Tensor,
    ):
        phi = self.mlp_phi(s)
        W = self.linear_W(radial_embeddings) * envelope[..., None]
        x = phi[:, None, ...] * W
        x_s, x_vv, x_vs = torch.split(x, self.num_features, dim=-1)
        ds = torch.sum(x_s, dim=2)  # sum over j
        x_v = v[:, None, ...] * x_vv + x_vs * unit_vectors[..., None]
        dv = torch.sum(x_v, dim=2)  # sum over j
        return s + ds, v + dv


class UpdateBlock(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        self.mlp_a = nn.Sequential(
            nn.Linear(num_features * 2, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features * 3),
        )
        self.linear_UV = nn.Linear(num_features, num_features * 2, bias=False)

    def forward(self, s: torch.Tensor, v: torch.Tensor):
        U_v, V_v = torch.split(self.linear_UV(v), self.num_features, dim=-1)
        V_v_norm = torch.sqrt(torch.sum(V_v ** 2, dim=-2, keepdim=True) + 1e-6)
        a = self.mlp_a(torch.cat((s, V_v_norm), dim=-1))
        a_vv, a_sv, a_ss = torch.split(a, self.num_features, dim=-1)
        dv = a_vv * U_v
        UV_dot = torch.sum(U_v * V_v, dim=-2, keepdim=True)
        ds = a_ss + a_sv * UV_dot
        return s + ds, v + dv


class EquivariantRMSNorm(nn.Module):
    def __init__(self, num_features: int, affine: bool = True, eps: float = 1e-6):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.affine_s_weight = nn.Parameter(torch.ones(num_features))
            self.affine_s_bias = nn.Parameter(torch.zeros(num_features))
            self.affine_v_weight = nn.Parameter(torch.ones(num_features))

    def forward(self, s: torch.Tensor, v: torch.Tensor):
        s = s - s.mean(dim=-1, keepdim=True)
        s = s / (s.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        v = v / (v.pow(2).mean(dim=(-2, -1), keepdim=True) + self.eps).sqrt()
        if self.affine:
            s = s * self.affine_s_weight + self.affine_s_bias
            v = v * self.affine_v_weight
        return s, v


class GatedEquivariantBlock(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_output_features: int,
        init_scale: float | None = None,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_output_features = num_output_features
        self.init_scale = init_scale
        self.linear_v1 = nn.Linear(num_features, num_features, bias=False)
        self.linear_v2 = nn.Linear(num_features, num_output_features, bias=False)
        self.mlp_s = nn.Sequential(
            nn.Linear(num_features * 2, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_output_features * 2),
        )
        self.act = nn.SiLU()

    def forward(self, s: torch.Tensor, v: torch.Tensor):
        W_v1_norm = torch.sqrt(
            torch.sum(self.linear_v1(v) ** 2, dim=-2, keepdim=True) + 1e-6
        )
        W_v2 = self.linear_v2(v)
        s_out, v_scale = torch.split(
            self.mlp_s(torch.cat((s, W_v1_norm), dim=-1)),
            self.num_output_features,
            dim=-1,
        )
        s_out = self.act(s_out)
        v_out = W_v2 * v_scale
        return s_out, v_out


class PotentialHead(nn.Module):
    def __init__(self, num_features: int, init_scale: float | None = None):
        super().__init__()
        self.init_scale = init_scale
        self.out_energy = nn.Sequential(
            nn.Linear(num_features, num_features // 2),
            nn.SiLU(),
            nn.Linear(num_features // 2, 1, bias=False),
        )

    def forward(self, s: torch.Tensor, v: torch.Tensor = None):
        return self.out_energy(s).squeeze(-1)  # [B, N, 1]


class VectorHead(nn.Module):
    def __init__(self, num_features: int, init_scale: float | None = None):
        super().__init__()
        self.out_vector = nn.ModuleList(
            [
                GatedEquivariantBlock(num_features, num_features // 2),
                GatedEquivariantBlock(num_features // 2, 1, init_scale=init_scale),
            ]
        )

    def forward(self, s: torch.Tensor, v: torch.Tensor):
        for block in self.out_vector:
            s, v = block(s, v)
        return v.squeeze(-1)  # [B, N, 3]


class PaiNN(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_radial_basis: int,
        num_layers: int,
        num_elements: int | None = None,
        r_max: float = 6.0,
        r_offset: float = 0.0,
        time_init_mode: Literal["node", "edge", "none"] = "node",
        parity_breaking: bool = True,
        conservative: bool = False,
        periodic_table: PeriodicTable | None = None,
        unique_atom_indices: bool = False,
        composition: str | None = None,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_radial_basis = num_radial_basis
        self.num_layers = num_layers
        self.r_max = r_max
        self.r_offset = r_offset
        self.r_cutoff = r_max + r_offset
        self.time_init_mode = time_init_mode
        self.conservative = conservative
        self.unique_atom_indices = unique_atom_indices

        self.atom_embedding = AtomEmbedding(num_features, num_elements, periodic_table)
        if time_init_mode == "edge":
            self.time_embedding = TimeEmbedding(num_radial_basis)
        elif time_init_mode == "node":
            self.time_embedding = TimeEmbedding(num_features)
        elif time_init_mode == "none":
            self.time_embedding = None
        else:
            raise ValueError(f"Invalid time embedding mode: {time_init_mode}")

        self.radial_embedding = RadialBasis(
            num_radial=num_radial_basis,
            cutoff=self.r_cutoff,
            rbf={"name": "gaussian"},
            envelope={"name": "polynomial", "exponent": 5},
        )

        messages, updates = [], []
        for _ in range(num_layers):
            if parity_breaking:
                messages.append(
                    ParityBreakingMessageBlock(num_features, num_radial_basis)
                )
            else:
                messages.append(MessageBlock(num_features, num_radial_basis))
            updates.append(UpdateBlock(num_features))
        self.messages = nn.ModuleList(messages)
        self.updates = nn.ModuleList(updates)

        norms_1, norms_2 = [], []
        for _ in range(num_layers):
            norms_1.append(EquivariantRMSNorm(num_features, affine=True))
            norms_2.append(EquivariantRMSNorm(num_features, affine=True))
        self.norms_1 = nn.ModuleList(norms_1)
        self.norms_2 = nn.ModuleList(norms_2)

        if conservative:
            self.output_block = PotentialHead(num_features)
        else:
            self.output_block = VectorHead(num_features)

        if composition is not None:
            z_list = composition_to_atomic_numbers(composition)
            self.register_buffer(
                "fixed_atomic_numbers", torch.tensor(z_list, dtype=torch.long)
            )
        else:
            self.fixed_atomic_numbers = None

    def forward(
        self,
        time: torch.Tensor,  # [B]
        pos: torch.Tensor,  # [B, N, 3]
        atomic_numbers: torch.Tensor | None = None,  # optional [B, N]
        return_potential: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        with torch.enable_grad() if self.conservative else nullcontext():
            if self.conservative:
                pos.requires_grad_(True)

            B, num_nodes, _ = pos.shape

            dist_vec = pos[:, :, None, :] - pos[:, None, :, :]
            r = torch.sqrt(torch.sum(dist_vec ** 2, dim=-1, keepdim=True) + 1e-6)
            unit_vec = dist_vec / r  # [B, N, N, 3]

            r_shifted = (r + self.r_offset).clamp(max=self.r_cutoff).squeeze(-1)
            radial_emb, env = self.radial_embedding(r_shifted, return_envelope=True)
            if self.time_init_mode == "edge":
                radial_emb = radial_emb + self.time_embedding(time[:, None, None])
            radial_emb = radial_emb[:, :, :, None, :]
            env = env[:, :, :, None]

            # Self-interaction mask
            mask = ~torch.eye(num_nodes, dtype=torch.bool, device=pos.device)
            mask = mask.view(1, num_nodes, num_nodes, 1).to(pos.dtype)
            unit_vec = unit_vec * mask
            env = env * mask

            if self.unique_atom_indices:
                atom_index = torch.arange(
                    num_nodes, dtype=torch.long, device=pos.device
                )
                atom_index = atom_index.unsqueeze(0).expand(B, -1)
                s = self.atom_embedding(atom_index)
            else:
                if atomic_numbers is not None:
                    z = atomic_numbers
                elif self.fixed_atomic_numbers is not None:
                    z = self.fixed_atomic_numbers.unsqueeze(0).expand(B, -1)
                else:
                    z = torch.zeros((B, num_nodes), dtype=torch.long, device=pos.device)
                s = self.atom_embedding(z)

            if self.time_init_mode == "node":
                s = s + self.time_embedding(time[:, None])
            s = s[:, :, None, :]  # [B, N, 1, n_feats]
            v = torch.zeros_like(s).repeat(1, 1, 3, 1)  # [B, N, 3, n_feats]

            for message, update, norm_1, norm_2 in zip(
                self.messages, self.updates, self.norms_1, self.norms_2
            ):
                s, v = message(s, v, radial_emb, env, unit_vec)
                s, v = norm_1(s, v)
                s, v = update(s, v)
                s, v = norm_2(s, v)

            output = self.output_block(s, v)  # [B, N, 3 or 1]

            if not self.conservative:
                return output  # [B, N, 3]

            potential_per_node = output.squeeze(-1)  # [B, N]
            potential_per_graph = potential_per_node.sum(dim=-1)  # [B]
            field = torch.autograd.grad(
                outputs=potential_per_graph,
                inputs=pos,
                grad_outputs=torch.ones_like(potential_per_graph),
                create_graph=self.training,
                retain_graph=self.training,
                allow_unused=True,
            )[0]
            if field is None:
                field = torch.zeros_like(pos)
            if return_potential:
                return field, potential_per_graph.detach()
            return field
