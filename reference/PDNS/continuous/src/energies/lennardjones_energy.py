from io import BytesIO
from typing import Optional
import os
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from bgflow import Energy
from bgflow.utils import distance_vectors, distances_from_vectors

from src.energies.base_energy_function import BaseEnergyFunction, BaseEnergy
from src.utils.graph_utils import remove_mean
from src.utils.common import download_github_file

from ipdb import set_trace as debug

def sample_from_array(array, size):
    idx = np.random.choice(array.shape[0], size=size)
    return array[idx]


def lennard_jones_energy_torch(r, eps=1.0, rm=1.0):
    p = 0.9
    lj = eps * ((rm / r) ** 12 - 2 * (rm / r) ** 6)
    return lj


class LennardJonesPotential(Energy):
    def __init__(
        self,
        dim,
        n_particles,
        eps=1.0,
        rm=1.0,
        oscillator=True,
        oscillator_scale=1.0,
        two_event_dims=True,
        energy_factor=1.0,
    ):
        """Energy for a Lennard-Jones cluster.

        Parameters
        ----------
        dim : int
            Number of degrees of freedom ( = space dimension x n_particles)
        n_particles : int
            Number of Lennard-Jones particles
        eps : float
            LJ well depth epsilon
        rm : float
            LJ well radius R_min
        oscillator : bool
            Whether to use a harmonic oscillator as an external force
        oscillator_scale : float
            Force constant of the harmonic oscillator energy
        two_event_dims : bool
            If True, the energy expects inputs with two event dimensions (particle_id, coordinate).
            Else, use only one event dimension.
        """
        if two_event_dims:
            super().__init__([n_particles, dim // n_particles])
        else:
            super().__init__(dim)
        self._n_particles = n_particles
        self._n_dims = dim // n_particles

        self._eps = eps
        self._rm = rm
        self.oscillator = oscillator
        self._oscillator_scale = oscillator_scale

        # this is to match the eacf energy with the eq-fm energy
        # for lj13, to match the eacf set energy_factor=0.5
        self._energy_factor = energy_factor

    def _energy(self, x):
        batch_shape = x.shape[: -len(self.event_shape)]
        x = x.view(*batch_shape, self._n_particles, self._n_dims)

        dists = distances_from_vectors(
            distance_vectors(x.view(-1, self._n_particles, self._n_dims))
        )

        lj_energies = lennard_jones_energy_torch(dists, self._eps, self._rm)
        # lj_energies = torch.clip(lj_energies, -1e4, 1e4)
        lj_energies = lj_energies.view(*batch_shape, -1).sum(dim=-1) * self._energy_factor

        if self.oscillator:
            osc_energies = 0.5 * self._remove_mean(x).pow(2).sum(dim=(-2, -1)).view(*batch_shape)
            lj_energies = lj_energies + osc_energies * self._oscillator_scale

        return lj_energies[:, None]

    def _remove_mean(self, x):
        x = x.view(-1, self._n_particles, self._n_dims)
        return x - torch.mean(x, dim=1, keepdim=True)

    def _energy_numpy(self, x):
        x = torch.Tensor(x)
        return self._energy(x).cpu().numpy()

    def _log_prob(self, x):
        return -self._energy(x)


class LennardJonesEnergy(BaseEnergyFunction, BaseEnergy):
    def __init__(
        self,
        name,
        dim,
        n_particles,
        data_path=None,
        data_path_train=None,
        data_path_val=None,
        data_path_test=None,
        device="cpu",
        plot_samples_epoch_period=5,
        plotting_buffer_sample_size=512,
        data_normalization_factor=1.0,
        energy_factor=1.0,
        is_molecule=True,
        **kwargs,
    ):
        
        self.n_particles = n_particles
        self.n_spatial_dim = dim // n_particles

        if self.n_particles != 13 and self.n_particles != 55:
            raise NotImplementedError

        self.curr_epoch = 0
        self.plotting_buffer_sample_size = plotting_buffer_sample_size
        self.plot_samples_epoch_period = plot_samples_epoch_period

        self.data_normalization_factor = data_normalization_factor

        if data_path is not None:
            download = lambda file: download_github_file(
                "jarridrb", "DEM", file
            )
            self.data_path = download(data_path)
            self.data_path_train = download(data_path_train)
            self.data_path_val = download(data_path_val)
            self.data_path_test = download(data_path_test)
        else:
            self.data_path = None
            self.data_path_train = None
            self.data_path_val = None
            self.data_path_test = None

        if self.n_particles == 13:
            name = "LJ13_efm"
        elif self.n_particles == 55:
            name = "LJ55"

        self.device = device

        self.lennard_jones = LennardJonesPotential(
            dim=dim,
            n_particles=n_particles,
            eps=1.0,
            rm=1.0,
            oscillator=True,
            oscillator_scale=1.0,
            two_event_dims=False,
            energy_factor=energy_factor,
        )

        BaseEnergyFunction.__init__(self, dim=dim, is_molecule=is_molecule)
        BaseEnergy.__init__(self, name, dim) # set self.name & self.dim

        

    def eval(self, samples: torch.Tensor) -> torch.Tensor:
        return - self.unnorm_log_prob(samples)

    def unnorm_log_prob(self, samples: torch.Tensor) -> torch.Tensor:
        return self.lennard_jones._log_prob(samples).squeeze(-1)

    def setup_test_set(self):
        if self.data_path_test is None:
            print("Data path for test data is not provided")
            return 0
        else:
            data = np.load(self.data_path_test, allow_pickle=True)
            data = remove_mean(data, self.n_particles, self.n_spatial_dim)
            data = torch.tensor(data, device=self.device)
            return data

    def setup_val_set(self):
        if self.data_path_val is None:
            print("Data path for validation data is not provided")
            return 0
        else:
            data = np.load(self.data_path_val, allow_pickle=True)
            data = remove_mean(data, self.n_particles, self.n_spatial_dim)
            data = torch.tensor(data, device=self.device)
            return data

    def setup_train_set(self):
        if self.data_path_train is None:
            print("Data path for training data is not provided")
            return 0
        else:
            if self.data_path_train.endswith(".pt"):
                data = torch.load(self.data_path_train).cpu().numpy()
            else:
                data = np.load(self.data_path_train, allow_pickle=True)

            data = remove_mean(data, self.n_particles, self.n_spatial_dim)
            data = torch.tensor(data, device=self.device)
            return data

    def interatomic_dist(self, x):
        batch_shape = x.shape[: -len(self.lennard_jones.event_shape)]
        x = x.view(*batch_shape, self.n_particles, self.n_spatial_dim)

        # Compute the pairwise interatomic distances
        # removes duplicates and diagonal
        distances = x[:, None, :, :] - x[:, :, None, :]
        distances = distances[
            :,
            torch.triu(torch.ones((self.n_particles, self.n_particles)), diagonal=1) == 1,
        ]
        dist = torch.linalg.norm(distances, dim=-1)
        return dist

    def to(self, device):
        self.lennard_jones.to(device)