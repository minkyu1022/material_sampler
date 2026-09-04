import torch
import numpy as np
import multiprocessing as mp

import mdtraj
from mdtraj.core.trajectory import Trajectory
from openmmtools.testsystems import AlanineDipeptideVacuum

from simtk import openmm as mm
from simtk.openmm import app
from simtk import unit

import PIL
import matplotlib as mpl
from matplotlib import pyplot as plt


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    # https://stackoverflow.com/a/61756899
    return PIL.Image.frombytes(
        'RGB',
        fig.canvas.get_width_height(),
        fig.canvas.tostring_rgb()
    )


# adapt from SNF
def get_Z_indices():
    cart_atom_indices = np.array([6, 8, 9, 10, 14])
    Z_indices_no_order = np.array([[0, 1, 4, 6],
                [1, 4, 6, 8],
                [2, 1, 4, 0],
                [3, 1, 4, 0],
                [4, 6, 8, 14],
                [5, 4, 6, 8],
                [7, 6, 8, 4],
                [11, 10, 8, 6],
                [12, 10, 8, 11],
                [13, 10, 8, 11],
                [15, 14, 8, 16],
                [16, 14, 8, 6],
                [17, 16, 14, 15],
                [18, 16, 14, 8],
                [19, 18, 16, 14],
                [20, 18, 16, 19],
                [21, 18, 16, 19]])

    cart_atom_indices = np.array(cart_atom_indices)
    # cart_indices = np.concatenate([[i*3, i*3+1, i*3+2] for i in cart_atom_indices])
    batchwise_Z_indices, _ = decompose_Z_indices(cart_atom_indices, Z_indices_no_order)
    Z_indices = np.vstack(batchwise_Z_indices)
    return Z_indices


def to_md_traj(traj: Trajectory | torch.Tensor, n_particles: int, spatial_dim: int) -> Trajectory:

    if isinstance(traj, torch.Tensor):
        # torch to Trajectory (used in both SNF and FAB)
        aldp = AlanineDipeptideVacuum()
        topology = mdtraj.Topology.from_openmm(aldp.topology)
        traj = mdtraj.Trajectory(traj.reshape(-1, n_particles, spatial_dim), topology)

    elif isinstance(traj, Trajectory):
        pass

    else:
        raise ValueError(f"Unsupported traj type: {type(traj)}")

    return traj


# adapt from SNF
def plot_ramachandran_AD(traj: Trajectory | torch.Tensor, weights: np.ndarray | None = None) -> PIL.Image:
    # config for AD
    n_particles = 22
    spatial_dim = 3

    traj = to_md_traj(traj, n_particles, spatial_dim)

    fig = plt.figure(figsize=(4, 4))
    ax = plt.gca()

    phi = mdtraj.compute_phi(traj)[1].reshape(-1)
    psi = mdtraj.compute_psi(traj)[1].reshape(-1)
    bins = [np.linspace(-np.pi, np.pi, 64), np.linspace(-np.pi, np.pi, 64)]
    ax.hist2d(phi, psi, bins=bins, norm=mpl.colors.LogNorm(), weights=weights)

    ax.set_xlim(-np.pi,np.pi)
    ax.set_ylim(-np.pi,np.pi)
    ax.set_xlabel(r'$\phi$')
    ax.set_ylabel(r'$\psi$')

    fig.canvas.draw()
    PIL_im = fig2img(fig)
    plt.close()
    return {"ramachandran": PIL_im}


# adapt from SNF
def periodic_convolution(x, kernel):
    x_padded = np.concatenate([x, x, x])
    y_padded = np.convolve(x_padded, kernel, mode='same')
    return y_padded[x.size:-x.size]


def compute_torsion_histogram(
    traj: Trajectory,
    Z_indices: np.ndarray,
    smooth: bool = False,
    nbins: int = 200,
    hist_range: list = [-np.pi, np.pi]
):
    # compute the dihedrals
    torsions = mdtraj.compute_dihedrals(traj, Z_indices)

    torsion_histogram = []
    for i in range(torsions.shape[1]):
        htrain, e = np.histogram(torsions[:, i], nbins, range=hist_range, density=True)
        if smooth:
            # smooth the histogram with a Gaussian kernel
            smooth_kernel = [0.25, 0.5, 1.0, 0.5, 0.25] # optional
            htrain = periodic_convolution(htrain, smooth_kernel)
        torsion_histogram.append(htrain)

    return torsion_histogram


def plot_energy_hist(gen_energies, ref_energies):
    fig = plt.figure(figsize=(4, 4))
    plt.hist(ref_energies, bins=30, alpha=0.5, label='ref', color='blue')
    plt.hist(gen_energies, bins=30, alpha=0.5, label='gen', color='orange')
    plt.legend(loc='upper right', fontsize=10)
    fig.canvas.draw()
    PIL_im = fig2img(fig)
    plt.close()
    return {"energy_hist": PIL_im}


# adapt from SNF
def plot_marginal_group_AD(
    gen_traj: Trajectory | torch.Tensor,
    ref_traj: Trajectory,
    smooth: bool = False,
    ref_torsion_hist=None,
) -> PIL.Image:
    # config for AD
    n_particles = 22
    spatial_dim = 3

    # plot config
    nbins = 200
    hist_range = [-np.pi, np.pi]
    torsions_simple = [4, 5, 6, 7, 8]
    torsions_complex = [0, 1, 2, 10, 12]

    # compute the histogram
    Z_indices = get_Z_indices()
    gen_traj = to_md_traj(gen_traj, n_particles, spatial_dim)
    if ref_torsion_hist is None:
        ref_torsion_hist = compute_torsion_histogram(ref_traj, Z_indices, smooth, nbins, hist_range)
    gen_torsion_hist = compute_torsion_histogram(gen_traj, Z_indices, smooth, nbins, hist_range)

    # plot the histogram
    fig, axes = plt.subplots(nrows=1, ncols=5, sharey=True, sharex=True, figsize=(15, 3))
    fig.subplots_adjust(hspace=0.05, wspace=0.15)
    titles = ['$\phi$', '$\gamma_1$', '$\psi$', '$\gamma_2$', '$\gamma_3$']
    for ax, torsion_idx, title in zip(axes, torsions_complex, titles):
        xticks = np.linspace(*hist_range, nbins)
        ax.plot(xticks, ref_torsion_hist[torsion_idx], color='grey', linewidth=3, label='ref')
        ax.plot(xticks, gen_torsion_hist[torsion_idx], color='blue', linewidth=3, label='gen')

        ax.set_xlim(-np.pi, np.pi)
        ax.set_xticks((-np.pi, -np.pi/2, 0, np.pi/2, np.pi))
        ax.set_xticklabels(('$-\pi$', '$-\pi/2$', 0, '$\pi/2$', '$\pi$'))

        ax.set_ylim(0, 1)
        ax.set_yticks((0, .25, .5, .75, 1))

        ax.set_title(title, fontsize=15)
        ax.grid(True)

    axes[0].set_ylabel('density', fontsize=15)
    axes[-1].legend(loc='upper right', fontsize=10)

    fig.canvas.draw()
    PIL_im = fig2img(fig)
    plt.close()

    # compute KL divergence
    EPS = 1e-5
    torsion_klds = []
    for torsion_idx in torsions_complex:
        ref_hist = ref_torsion_hist[torsion_idx]
        gen_hist = gen_torsion_hist[torsion_idx]

        kld_unscaled = np.sum(ref_hist * np.log((ref_hist + EPS) / (gen_hist + EPS)))
        kld = kld_unscaled * (hist_range[1] - hist_range[0]) / nbins
        torsion_klds.append(kld)

    return {
        "marginal": PIL_im,
        "kld_phi": torsion_klds[0],
        "kld_gamma1": torsion_klds[1],
        "kld_psi": torsion_klds[2],
        "kld_gamma2": torsion_klds[3],
        "kld_gamma3": torsion_klds[4],
    }


# ====================================================
# ==== copy from bgtorch.nn.flow.crd_transform.py ====
# ====================================================


def decompose_Z_indices(cart_indices, Z_indices):
    import numpy as np
    known_indices = cart_indices
    Z_placed = np.zeros(Z_indices.shape[0])
    Z_indices_decomposed = []
    while np.count_nonzero(Z_placed) < Z_indices.shape[0]:
        Z_indices_cur = []
        for i in range(Z_indices.shape[0]):
            if not Z_placed[i] and np.all([Z_indices[i, j] in known_indices for j in range(1, 4)]):
                Z_indices_cur.append(Z_indices[i])
                Z_placed[i] = 1
        Z_indices_cur = np.array(Z_indices_cur)
        known_indices = np.concatenate([known_indices, Z_indices_cur[:, 0]])
        Z_indices_decomposed.append(Z_indices_cur)

    index2order = np.concatenate([cart_indices] + [Z[:, 0] for Z in Z_indices_decomposed])

    return Z_indices_decomposed, index2order


# ====================================================
# ====== copy from boltzgen/openmm_interface.py ======
# ====================================================


# Gas constant in kJ / mol / K
R = 8.314e-3


class OpenMMEnergyInterface(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, openmm_context, temperature):
        device = input.device
        n_batch = input.shape[0]
        input = input.view(n_batch, -1, 3)
        n_dim = input.shape[1]
        energies = torch.zeros((n_batch, 1), dtype=input.dtype)
        forces = torch.zeros_like(input)

        kBT = R * temperature
        input = input.cpu().detach().numpy()
        for i in range(n_batch):
            # reshape the coordinates and send to OpenMM
            x = input[i, :].reshape(-1, 3)
            # Handle nans and infinities
            if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                energies[i, 0] = np.nan
            else:
                openmm_context.setPositions(x)
                state = openmm_context.getState(getForces=True, getEnergy=True)

                # get energy
                energies[i, 0] = (
                    state.getPotentialEnergy().value_in_unit(
                        unit.kilojoule / unit.mole) / kBT
                )

                # get forces
                f = (
                    state.getForces(asNumpy=True).value_in_unit(
                        unit.kilojoule / unit.mole / unit.nanometer
                    )
                    / kBT
                )
                forces[i, :] = torch.from_numpy(-f)
        forces = forces.view(n_batch, n_dim * 3)
        # Save the forces for the backward step, uploading to the gpu if needed
        ctx.save_for_backward(forces.to(device=device))
        return energies.to(device=device)

    @staticmethod
    def backward(ctx, grad_output):
        forces, = ctx.saved_tensors
        return forces * grad_output, None, None


class OpenMMEnergyInterfaceParallel(torch.autograd.Function):
    """
    Uses parallel processing to get the energies of the batch of states
    """
    @staticmethod
    def var_init(sys, temp):
        """
        Method to initialize temperature and openmm context for workers
        of multiprocessing pool
        """
        global temperature, openmm_context
        temperature = temp
        sim = app.Simulation(sys.topology, sys.system,
                             mm.LangevinIntegrator(temp * unit.kelvin,
                                                   1.0 / unit.picosecond,
                                                   1.0 * unit.femtosecond),
                             platform=mm.Platform.getPlatformByName('Reference'))
        openmm_context = sim.context

    @staticmethod
    def batch_proc(input):
        # Process state
        # openmm context and temperature are passed a global variables
        input = input.reshape(-1, 3)
        n_dim = input.shape[0]

        kBT = R * temperature
        # Handle nans and infinities
        if np.any(np.isnan(input)) or np.any(np.isinf(input)):
            energy = np.nan
            force = np.zeros_like(input)
        else:
            openmm_context.setPositions(input)
            state = openmm_context.getState(getForces=True, getEnergy=True)

            # get energy
            energy = state.getPotentialEnergy().value_in_unit(
                unit.kilojoule / unit.mole) / kBT

            # get forces
            force = -state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule / unit.mole / unit.nanometer) / kBT
        force = force.reshape(n_dim * 3)
        return energy, force

    @staticmethod
    def forward(ctx, input, pool):
        device = input.device
        input_np = input.cpu().detach().numpy()
        energies_out, forces_out = zip(*pool.map(
            OpenMMEnergyInterfaceParallel.batch_proc, input_np))
        energies_np = np.array(energies_out)[:, None]
        forces_np = np.array(forces_out)
        energies = torch.from_numpy(energies_np)
        forces = torch.from_numpy(forces_np)
        energies = energies.type(input.dtype)
        forces = forces.type(input.dtype)
        # Save the forces for the backward step, uploading to the gpu if needed
        ctx.save_for_backward(forces.to(device=device))
        return energies.to(device=device)

    @staticmethod
    def backward(ctx, grad_output):
        forces, = ctx.saved_tensors
        return forces * grad_output, None, None


def regularize_energy(energy, energy_cut, energy_max):
    # Cast inputs to same type
    energy_cut = energy_cut.type(energy.type())
    energy_max = energy_max.type(energy.type())
    # Check whether energy finite
    energy_finite = torch.isfinite(energy)
    # Cap the energy at energy_max
    energy = torch.where(energy < energy_max, energy, energy_max)
    # Make it logarithmic above energy cut and linear below
    energy = torch.where(
        energy < energy_cut, energy, torch.log(energy - energy_cut + 1) + energy_cut
    )
    energy = torch.where(energy_finite, energy,
                         torch.tensor(np.nan, dtype=energy.dtype, device=energy.device))
    return energy


# ====================================================
# ======== copy from boltzgen/distribution.py ========
# ====================================================


class Boltzmann:
    """
    Boltzmann distribution using OpenMM to get energy and forces
    """
    def __init__(self, sim_context, temperature, energy_cut, energy_max):
        """
        Constructor
        :param sim_context: Context of the simulation object used for energy
        and force calculation
        :param temperature: Temperature of System
        """
        # Save input parameters
        self.sim_context = sim_context
        self.temperature = temperature
        self.energy_cut = torch.tensor(energy_cut)
        self.energy_max = torch.tensor(energy_max)

        # Set up functions
        self.openmm_energy = OpenMMEnergyInterface.apply
        self.regularize_energy = regularize_energy

        self.norm_energy = lambda pos: self.regularize_energy(
            self.openmm_energy(pos, self.sim_context, temperature)[:, 0],
            self.energy_cut, self.energy_max)

    def log_prob(self, z):
        return -self.norm_energy(z)


class BoltzmannParallel:
    """
    Boltzmann distribution using OpenMM to get energy and forces and processes the
    batch of states in parallel
    """
    def __init__(self, system, temperature, energy_cut, energy_max, n_threads=None):
        """
        Constructor
        :param system: Molecular system
        :param temperature: Temperature of System
        :param energy_cut: Energy at which logarithm is applied
        :param energy_max: Maximum energy
        :param n_threads: Number of threads to use to process batches, set
        to the number of cpus if None
        """
        # Save input parameters
        self.system = system
        self.temperature = temperature
        self.energy_cut = torch.tensor(energy_cut)
        self.energy_max = torch.tensor(energy_max)
        self.n_threads = mp.cpu_count() if n_threads is None else n_threads

        # Create pool for parallel processing
        self.pool = mp.Pool(self.n_threads, OpenMMEnergyInterfaceParallel.var_init,
                            (system, temperature))

        # Set up functions
        self.openmm_energy = OpenMMEnergyInterfaceParallel.apply
        self.regularize_energy = regularize_energy

        self.norm_energy = lambda pos: self.regularize_energy(
            self.openmm_energy(pos, self.pool)[:, 0],
            self.energy_cut, self.energy_max)

    def log_prob(self, z):
        return -self.norm_energy(z)
