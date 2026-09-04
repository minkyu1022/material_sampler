import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from bgflow import Energy
from src.energies.base_energy_function import BaseEnergyFunction, BaseEnergy
# from dem.utils.logging_utils import fig2img
# from bgflow import MultiDoubleWellPotential

from src.utils.graph_utils import remove_mean


class MultiDoubleWellPotential(Energy):
    """Energy for a many particle system with pair wise double-well interactions.
    The energy of the double-well is given via

    .. math::
        E_{DW}(d) = a \cdot (d-d_{\text{offset})^4 + b \cdot (d-d_{\text{offset})^2 + c.

    Parameters
    ----------
    dim : int
        Number of degrees of freedom ( = space dimension x n_particles)
    n_particles : int
        Number of particles
    a, b, c, offset : float
        parameters of the potential
    """

    def __init__(self, dim, n_particles, a, b, c, offset, two_event_dims=True):
        if two_event_dims:
            super().__init__([n_particles, dim // n_particles])
        else:
            super().__init__(dim)
        self._dim = dim
        self._n_particles = n_particles
        self._spatial_dim = dim // n_particles
        self._a = a
        self._b = b
        self._c = c
        self._offset = offset

    def _energy(self, x):
        x = x.contiguous()
        dists = compute_distances(x, self._n_particles, self._spatial_dim)
        dists = dists - self._offset

        energies = self._a * dists ** 4 + self._b * dists ** 2 + self._c
        return energies.sum(-1, keepdim=True)


# TODO(ghliu) should only inherit BaseEnergy
class MultiDoubleWellEnergy(BaseEnergyFunction, BaseEnergy):
    def __init__(
        self,
        dim,
        n_particles,
        data_path,
        data_path_train=None,
        data_path_val=None,
        data_path_test=None,
        data_from_efm=True,  # if False, data from EFM
        device="cpu",
        plot_samples_epoch_period=5,
        plotting_buffer_sample_size=512,
        data_normalization_factor=1.0,
        is_molecule=True,
        **kwargs,
    ):
        
        self.n_particles = n_particles
        self.n_spatial_dim = dim // n_particles

        self.curr_epoch = 0
        self.plotting_buffer_sample_size = plotting_buffer_sample_size
        self.plot_samples_epoch_period = plot_samples_epoch_period

        self.data_normalization_factor = data_normalization_factor

        self.data_from_efm = data_from_efm

        if data_from_efm:
            name = "DW4_EFM"
        else:
            name = "DW4_EACF"

        if self.data_from_efm:
            if data_path_train is None:
                raise ValueError("DW4 is from EFM. No train data path provided")
            if data_path_val is None:
                raise ValueError("DW4 is from EFM. No val data path provided")

        # self.data_path = get_original_cwd() + "/" + data_path
        # self.data_path_train = get_original_cwd() + "/" + data_path_train
        # self.data_path_val = get_original_cwd() + "/" + data_path_val
        
        parent = '../../../'
        self.data_path = os.path.join(parent, data_path)
        self.data_path_train = os.path.join(parent, data_path_train)
        self.data_path_val = os.path.join(parent, data_path_val)
        self.data_path_test = os.path.join(parent, data_path_test)

        self.device = device

        self.val_set_size = 1000
        self.test_set_size = 1000
        self.train_set_size = 100000

        self.multi_double_well = MultiDoubleWellPotential(
            dim=dim,
            n_particles=n_particles,
            a=0.9,
            b=-4,
            c=0,
            offset=4,
            two_event_dims=False,
        )

        BaseEnergyFunction.__init__(self, dim=dim, is_molecule=is_molecule)
        BaseEnergy.__init__(self, name, dim) # set self.name & self.dim


    def eval(self, sample: torch.Tensor) -> torch.Tensor:
        return - self.unnorm_log_prob(sample)

    def unnorm_log_prob(self, samples: torch.Tensor) -> torch.Tensor:
        return - self.multi_double_well._energy(samples).squeeze(-1)

    def setup_test_set(self):
        if self.data_from_efm:
            data = np.load(self.data_path, allow_pickle=True)

        else:
            all_data = np.load(self.data_path, allow_pickle=True)
            data = all_data[0][-self.test_set_size :]
            del all_data

        data = remove_mean(torch.tensor(data), self.n_particles, self.n_spatial_dim).to(
            self.device
        )

        return data

    def setup_train_set(self):
        if self.data_from_efm:
            data = np.load(self.data_path_train, allow_pickle=True)

        else:
            all_data = np.load(self.data_path, allow_pickle=True)
            data = all_data[0][: self.train_set_size]
            del all_data

        data = remove_mean(torch.tensor(data), self.n_particles, self.n_spatial_dim).to(
            self.device
        )

        return data

    def setup_val_set(self):
        if self.data_from_efm:
            data = np.load(self.data_path_val, allow_pickle=True)

        else:
            all_data = np.load(self.data_path, allow_pickle=True)
            data = all_data[0][-self.test_set_size - self.val_set_size : -self.test_set_size]
            del all_data

        data = remove_mean(torch.tensor(data), self.n_particles, self.n_spatial_dim).to(
            self.device
        )
        return data

    def interatomic_dist(self, x):
        batch_shape = x.shape[: -len(self.multi_double_well.event_shape)]
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
        self.multi_double_well.to(device)


def compute_distances(x, n_particles, spatial_dim, remove_duplicates=True):
    """
    Computes the all distances for a given particle configuration x.

    Parameters
    ----------
    x : torch.Tensor
        Positions of n_particles in spatial_dim.
    remove_duplicates : boolean
        Flag indicating whether to remove duplicate distances
        and distances be.
        If False the all distance matrix is returned instead.

    Returns
    -------
    distances : torch.Tensor
        All-distances between particles in a configuration
        Tensor of shape `[n_batch, n_particles * (n_particles - 1) // 2]` if remove_duplicates.
        Otherwise `[n_batch, n_particles , n_particles]`
    """
    x = x.reshape(-1, n_particles, spatial_dim)
    distances = torch.cdist(x, x)
    # diff = x[:,:,None,:] - x[:,None,:,:]  # Shape: (batch, n_particles, n_particles, spatial_dim)
    # distances = torch.sqrt(torch.sum(diff**2, axis=-1) + 1e-8)  # Shape: (batch, n_particles, n_particles)
    if remove_duplicates:
        distances = distances[:, torch.triu(torch.ones((n_particles, n_particles)), diagonal=1) == 1]
        distances = distances.reshape(-1, n_particles * (n_particles - 1) // 2)
    return distances
