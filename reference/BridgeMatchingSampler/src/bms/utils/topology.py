# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Helpers for reading/writing molecular topologies and trajectories."""

import mdtraj as md
import numpy as np
import torch


def pdb_to_bond_indices(pdb_file: str) -> np.ndarray:
    """Extract bond indices from a PDB file."""
    pdb = md.load(pdb_file)
    bond_indices = []
    for bond in pdb.topology.bonds:
        bond_indices.append([bond[0].index, bond[1].index])
    return np.array(bond_indices).T


def save_data_to_pdb(data, topology_file: str, output_file: str):
    """Save positions to a PDB file, using the topology from a reference PDB file.

    Args:
        data: Positions of shape ``(B, N, 3)`` in units of Angstrom. Accepts either
            a ``torch.Tensor`` or a ``numpy.ndarray``.
        topology_file: Reference PDB file providing the topology.
        output_file: Destination PDB file.
    """
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    topology = md.load(topology_file).topology
    data = data.reshape(-1, topology.n_atoms, 3)
    traj = md.Trajectory(data / 10.0, topology=topology)  # Angstrom -> nm
    traj.save(output_file)
