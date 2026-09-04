import os
import torch
from pathlib import Path

from simtk import openmm as mm
from simtk import unit
from simtk.openmm import app
from openmmtools import testsystems

import mdtraj

from src.energies.base_energy_function import BaseEnergyFunction, BaseEnergy
from src.utils.md_utils import Boltzmann, BoltzmannParallel
from src.utils.graph_utils import remove_mean

from fab.target_distributions.aldp import AldpBoltzmann


# from FAB
def make_aldp_model(config, device):

    # Target distribution
    transform_mode = 'mixed' if not 'transform' in config['system'] \
        else config['system']['transform']
    shift_dih = False if not 'shift_dih' in config['system'] \
        else config['system']['shift_dih']
    env = 'vacuum' if not 'env' in config['system'] \
        else config['system']['env']
    ind_circ_dih = [0, 1, 2, 3, 4, 5, 8, 9, 10, 13, 15, 16]
    target = AldpBoltzmann(data_path=config['data']['transform'],
                           temperature=config['system']['temperature'],
                           energy_cut=config['system']['energy_cut'],
                           energy_max=config['system']['energy_max'],
                           n_threads=config['system']['n_threads'],
                           transform=transform_mode,
                           ind_circ_dih=ind_circ_dih,
                           shift_dih=shift_dih,
                           env=env)
    target = target.to(device)
    return target


def load_config(path):
    import yaml
    with open(path, 'r') as stream:
        return yaml.load(stream, yaml.FullLoader)

# AD energy function with internal coordainte
class AlanineDipeptideEnergyIC(BaseEnergyFunction, BaseEnergy):
    def __init__(
        self,
        name,
        dim,
        n_particles,
        data_path_val=None,
        data_path_test=None,
        device="cpu",
        is_molecule=True,
        # FAB config
        # temperature=300,
        # energy_cut=1e8,
        # energy_max=1e20,
        # n_threads=18,
        # env='implicit',
        **kwargs,
):
        """
        Boltzmann distribution of Alanine dipeptide
        :param temperature: Temperature of the system
        :type temperature: Integer
        :param energy_cut: Value after which the energy is logarithmically scaled
        :type energy_cut: Float
        :param energy_max: Maximum energy allowed, higher energies are cut
        :type energy_max: Float
        :param n_threads: Number of threads used to evaluate the log
            probability for batches
        :type n_threads: Integer
        """
        self.n_particles = 22
        self.n_spatial_dim = 3
        self.data_path_val = data_path_val
        self.data_path_test = data_path_test
        self.device = device

        # import urllib
        # urllib.request.urlretrieve(
        #     'https://huggingface.co/VincentStimper/fab/resolve/main/aldp/config.yaml',
        #     '/private/home/ghliu/code/adjoint-sb-sampler/fab/config.yaml'
        # )

        repo_dir = Path(__file__).resolve().parent.parent.parent

        config = load_config(repo_dir / 'fab/config.yaml')
        config['data']['transform'] = str(repo_dir / "fab/position_min_energy.pt")
        self.p = make_aldp_model(config, device)
        self.p.float() # NOTE(ghliu) convert to float32. FAB uses float64?

        BaseEnergy.__init__(self, name, dim) # set self.name & self.dim
        BaseEnergyFunction.__init__(self, dim=dim, is_molecule=is_molecule)

    def x_to_z(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self.p.coordinate_transform.inverse(x)
        return z

    def z_to_x(self, z: torch.Tensor) -> torch.Tensor:
        x, _ = self.p.coordinate_transform.forward(z)
        return x

    def log_prob_x(self, z: torch.tensor):
        return - self.p.p.norm_energy(z)

    def log_prob(self, z: torch.tensor):
        # z_, log_det = self.transform(z)
        # return -self.norm_energy(z_) + log_det
        return self.p.log_prob(z)

    def eval(self, z: torch.Tensor) -> torch.Tensor:
        return - self.log_prob(z)

    def eval_x(self, x: torch.Tensor) -> torch.Tensor:
        return - self.log_prob_x(x)

    # def unnorm_log_prob(self, z: torch.Tensor) -> torch.Tensor:
    #     return self.log_prob(z)

    def load_data(self, data_path):
        current_file_path = Path(os.path.abspath(__file__))
        data_path = current_file_path.parent.parent.parent / Path(data_path)

        data = mdtraj.load(data_path)
        # data = torch.from_numpy(data.xyz).reshape(-1, self.dim)

        # data = remove_mean(data, self.n_particles, self.n_spatial_dim)
        return data

    def setup_test_set(self):
        if self.data_path_test is not None:
            print("Loading test set ...")
            return self.load_data(self.data_path_test)

    def setup_val_set(self):
        if self.data_path_val is not None:
            print("Loading val set ...")
            return self.load_data(self.data_path_val)

    def interatomic_dist(self, x):
        batch_shape = x.shape[:-1]
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
        pass


class AlanineDipeptideEnergyICBounded(AlanineDipeptideEnergyIC):
    """
    AD IC Energy with hard boundaries on angle ranges
    Gives probability near 0 when angles are outside one cycle
    """
    def __init__(
        self,
        name,
        dim,
        n_particles,
        data_path_val=None,
        data_path_test=None,
        device="cpu",
        is_molecule=True,
        bond_length_bounds=(0.5, 3.0),  # Angstroms
        bond_angle_bounds=(0.1, 3.0),   # radians (avoid 0 and π)
        dihedral_bounds=(-3.14159, 3.14159),  # radians [-π, π]
        boundary_penalty=1000.0,  # Large penalty for out-of-bounds
        **kwargs
    ):
        super().__init__(
            name=name,
            dim=dim,
            n_particles=n_particles,
            data_path_val=data_path_val,
            data_path_test=data_path_test,
            device=device,
            is_molecule=is_molecule,
            **kwargs
        )
        
        # Store boundary parameters
        self.bond_length_bounds = bond_length_bounds
        self.bond_angle_bounds = bond_angle_bounds  
        self.dihedral_bounds = dihedral_bounds
        self.boundary_penalty = boundary_penalty
        
        # Define coordinate indices based on Z-matrix structure
        # For 19 Z-matrix atoms, each with 3 internal coordinates:
        # [9 Cartesian] + [19 bonds] + [19 angles] + [19 dihedrals] = 66 total
        # But we have 60D, so it's: [9 Cartesian] + [17 bonds] + [17 angles] + [17 dihedrals] = 60
        self.cart_indices = torch.arange(9, device=device)  # First 9 are Cartesian
        self.bond_indices = torch.arange(9, 26, device=device)   # Next 17 are bond lengths
        self.angle_indices = torch.arange(26, 43, device=device)  # Next 17 are bond angles  
        self.dihedral_indices = torch.arange(43, 60, device=device)  # Last 17 are dihedrals
        
    def _check_boundaries(self, z: torch.Tensor) -> torch.Tensor:
        """
        Check if internal coordinates are within bounds
        Returns penalty energy for out-of-bounds values
        """
        batch_size = z.shape[0]
        penalty = torch.zeros(batch_size, device=z.device, dtype=z.dtype)
        
        # Skip boundary checking if penalty is 0 (disabled)
        if self.boundary_penalty == 0.0:
            return penalty
        
        # Check bond length bounds
        if len(self.bond_indices) > 0:
            bond_lengths = z[:, self.bond_indices]
            bond_out_of_bounds = (
                (bond_lengths < self.bond_length_bounds[0]) | 
                (bond_lengths > self.bond_length_bounds[1])
            )
            penalty += bond_out_of_bounds.float().sum(dim=1) * self.boundary_penalty
        
        # Check bond angle bounds  
        if len(self.angle_indices) > 0:
            bond_angles = z[:, self.angle_indices]
            angle_out_of_bounds = (
                (bond_angles < self.bond_angle_bounds[0]) | 
                (bond_angles > self.bond_angle_bounds[1])
            )
            penalty += angle_out_of_bounds.float().sum(dim=1) * self.boundary_penalty
        
        # Check dihedral bounds
        if len(self.dihedral_indices) > 0:
            dihedrals = z[:, self.dihedral_indices]
            dihedral_out_of_bounds = (
                (dihedrals < self.dihedral_bounds[0]) | 
                (dihedrals > self.dihedral_bounds[1])
            )
            penalty += dihedral_out_of_bounds.float().sum(dim=1) * self.boundary_penalty
        
        return penalty
        
    def log_prob(self, z: torch.tensor):
        """Override to add boundary penalties"""
        try:
            # Get original log probability
            original_log_prob = super().log_prob(z)
            
            # Add boundary penalty
            boundary_penalty = self._check_boundaries(z)
            
            # Return penalized log probability
            # Use clamp to prevent extreme values that could cause NaN
            penalized_log_prob = original_log_prob - boundary_penalty
            return torch.clamp(penalized_log_prob, min=-1e6, max=1e6)
            
        except Exception as e:
            print(f"Error in bounded log_prob: {e}")
            # Fallback to original implementation
            return super().log_prob(z)
        
    def eval(self, z: torch.Tensor) -> torch.Tensor:
        """Override to add boundary penalties"""
        try:
            # Get original energy
            original_energy = super().eval(z)
            
            # Add boundary penalty  
            boundary_penalty = self._check_boundaries(z)
            
            # Return penalized energy
            # Use clamp to prevent extreme values that could cause NaN
            penalized_energy = original_energy + boundary_penalty
            return torch.clamp(penalized_energy, min=-1e6, max=1e6)
            
        except Exception as e:
            print(f"Error in bounded eval: {e}")
            # Fallback to original implementation
            return super().eval(z)
    
    def to(self, device):
        """Override to handle device transfers properly"""
        super().to(device)
        self.device = device
        # Update indices to new device
        self.cart_indices = self.cart_indices.to(device)
        self.bond_indices = self.bond_indices.to(device)
        self.angle_indices = self.angle_indices.to(device)
        self.dihedral_indices = self.dihedral_indices.to(device)
        return self


# FAB implementation, adapted FAB config
class AlanineDipeptideEnergy(BaseEnergyFunction, BaseEnergy):
    def __init__(
        self,
        name,
        dim,
        n_particles,
        data_path_val=None,
        data_path_test=None,
        device="cpu",
        is_molecule=True,
        # FAB config
        temperature=300,
        energy_cut=1e8,
        energy_max=1e20,
        n_threads=18,
        env='implicit',
        **kwargs,
):
        """
        Boltzmann distribution of Alanine dipeptide
        :param temperature: Temperature of the system
        :type temperature: Integer
        :param energy_cut: Value after which the energy is logarithmically scaled
        :type energy_cut: Float
        :param energy_max: Maximum energy allowed, higher energies are cut
        :type energy_max: Float
        :param n_threads: Number of threads used to evaluate the log
            probability for batches
        :type n_threads: Integer
        """
        self.n_particles = n_particles
        self.n_spatial_dim = dim // n_particles
        self.data_path_val = data_path_val
        self.data_path_test = data_path_test
        self.device = device

        # Define molecule parameters
        self.temperature = temperature
        self.energy_cut = energy_cut
        self.energy_max = energy_max

        if env == 'vacuum':
            system = testsystems.AlanineDipeptideVacuum(constraints=None)
        elif env == 'implicit':
            system = testsystems.AlanineDipeptideImplicit(constraints=None)
        else:
            raise NotImplementedError('This environment is not implemented.')

        sim = app.Simulation(system.topology, system.system,
                             mm.LangevinIntegrator(temperature * unit.kelvin,
                                                   1. / unit.picosecond,
                                                   1. * unit.femtosecond),
                             mm.Platform.getPlatformByName('Reference'))

        if n_threads > 1:
            self.p = BoltzmannParallel(
                system,
                temperature,
                energy_cut=energy_cut,
                energy_max=energy_max,
                n_threads=n_threads
            )
        else:
            self.p = Boltzmann(sim.context,
                temperature,
                energy_cut=energy_cut,
                energy_max=energy_max,
            )

        BaseEnergy.__init__(self, name, dim) # set self.name & self.dim
        BaseEnergyFunction.__init__(self, dim=dim, is_molecule=is_molecule)

    def log_prob(self, x: torch.tensor):
        return self.p.log_prob(x)

    def eval(self, samples: torch.Tensor) -> torch.Tensor:
        return - self.log_prob(samples)

    def unnorm_log_prob(self, samples: torch.Tensor) -> torch.Tensor:
        return self.log_prob(samples)

    def load_data(self, data_path):
        current_file_path = Path(os.path.abspath(__file__))
        data_path = current_file_path.parent.parent.parent / Path(data_path)

        data = mdtraj.load(data_path)
        # data = torch.from_numpy(data.xyz).reshape(-1, self.dim)

        # data = remove_mean(data, self.n_particles, self.n_spatial_dim)
        return data

    def setup_test_set(self):
        if self.data_path_test is not None:
            print("Loading test set ...")
            return self.load_data(self.data_path_test)

    def setup_val_set(self):
        if self.data_path_val is not None:
            print("Loading val set ...")
            return self.load_data(self.data_path_val)

    def interatomic_dist(self, x):
        batch_shape = x.shape[:-1]
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
        pass
