"""
Utility functions for the 2D Ising models

Only sampling algorithms are implemented by numpy, others are implemented by torch.

Please be aware of the input shape and range before use!
The check of dimension and range are all disabled for efficiency.
"""


from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


def ising2d_ham(S, J=1.0, h=0.0):
    r"""
    Compute the Hamiltonian for a batch of configurations in a 2D Ising model, with periodic boundary conditions.
    
    Parameters:
    - S: torch.tensor of shape (B, L * L): each element is -1 or 1, representing spin configurations.
    - J: float, interaction strength between neighboring spins (default=1.0).
    - h: float, external magnetic field strength (default=0.0).

    Returns:
    - hamiltonians: torch.tensor of shape (B,) containing the Hamiltonian for each configuration.
        H = -J \sum_{i \sim j} S_{i} S_{j} - h \sum_{i} S_{i}
    (The p.m.f. is given by p(S) \propto e^{-\beta H(S)})
    """
    # assert torch.all((S == 1) | (S == -1)), "All entries of S must be either 1 or -1"
    # be careful, we have disabled the verification of the range for efficiency!
    S = S.view(S.size(0), int(S.shape[1]**.5), int(S.shape[1]**.5))
    Sx = torch.roll(S, shifts=-1, dims=1)  # Sx[i,j] = S[i+1,j]
    Sy = torch.roll(S, shifts=-1, dims=2)  # Sy[i,j] = S[i,j+1]
    interaction_energy = -float(J) * torch.sum(S * (Sx + Sy), dim=(1, 2))
    magnetic_energy = -float(h) * torch.sum(S, dim=(1, 2))
    return interaction_energy + magnetic_energy


def ising2d_get_all_configs(L=4, device='cuda:0'):
    """
    Generate all possible Ising configurations for L x L lattice in increasing order
    e.g., [-1, -1], [-1, 1], [1, -1], [1, 1].
    Return: [2 ** (L ** 2), L ** 2], values are in {1, -1}
    """
    B = 2 ** (L ** 2)
    bits = torch.arange(L ** 2 - 1, -1, -1, device=device)
    return (((torch.arange(B, device=device)[:, None] >> bits) & 1) * 2 - 1).to(torch.int8) # [B, L ** 2]


def ising2d_mag(S):
    """
    Compute the magnetization for a batch of configurations.

    Parameters:
    S: torch.tensor of shape (B, L * L) representing B configurations on an L x L lattice.
       Each element in S is +1 or -1.

    Returns:
    - a float: average magnetization over the batch.
    """
    # assert torch.all((S == 1) | (S == -1)), "All entries of S must be either 1 or -1"
    return S.float().mean().item()


def ising2d_2pt_corr(S, r):
    """
    Compute the two-point correlation function for a batch of configurations.
    
    Parameters:
    - S: torch.tensor of shape (B, L * L) representing K configurations on an L x L lattice.
         Each element in S is +1 or -1.
    - r: int, horizontal and vertical distance between points for correlation calculation.
    
    Return:
    - a float: average two-point correlation at distance r over the batch.
    """
    # assert torch.all((S == 1) | (S == -1)), "All entries of S must be either 1 or -1"
    S = S.view(S.size(0), int(S.shape[1]**.5), int(S.shape[1]**.5)).to(dtype=torch.float)
    Sx = torch.roll(S, shifts=-r, dims=1)
    Sy = torch.roll(S, shifts=-r, dims=2)
    corr_x = ((S * Sx).mean(dim=0) - S.mean(dim=0) * Sx.mean(dim=0)).mean().item()
    corr_y = ((S * Sy).mean(dim=0) - S.mean(dim=0) * Sy.mean(dim=0)).mean().item()
    return (corr_x + corr_y) / 2


def ising2d_plot_2pt_corr(S):
    """
    Plot the 2-point correlation function for a batch of Ising configurations.

    Args:
        S: 2D tensor of shape (B, D=L^2) representing N Ising configurations, values in {-1, 1}.
    """
    L = int(S.shape[1]**0.5)
    # assert torch.all((S == 1) | (S == -1)), "All entries of S must be either 1 or -1"

    plt.close('all')
    fig = plt.figure()
    r = np.arange(-L//2, L//2 + 1)
    corr = [ising2d_2pt_corr(S, i) for i in r]
    plt.plot(r, corr, marker='o')
    plt.xlabel('Distance $r$')
    plt.xticks(r, [f'${i}$' for i in r])
    plt.ylim(-0.05, 1.05)
    plt.ylabel('2-point Correlation')
    return fig


def ising2d_emp_dist(samples):
    """
    samples: [B, L^2], elements in {-1, 1}
    Output the empirical distribution of the samples as a probability vector of length 2^{L^2}
    The configuations are sorted in increasing order 
    """
    # assert torch.all((samples == 1) | (samples == -1)), "All entries of samples must be either 1 or -1"
    B, N = samples.shape  # N = L^2
    bin_samples = ((samples + 1) // 2).to(torch.int32)  # (B, N)
    bits = torch.arange(N - 1, -1, -1, device=samples.device)
    indices = (bin_samples << bits).sum(dim=1)  # (B,)
    counts = torch.bincount(indices, minlength=2 ** N)
    return counts.float() / B


def ising2d_get_par_func(L, J=1.0, h=0.0, beta=1.0, apply_log=False, device='cuda:0'):
    """
    Compute the (log) partition function of a 2D Ising model with periodic boundary conditions.
    """
    all_configs = ising2d_get_all_configs(L, device)
    log_pmf = -beta * ising2d_ham(all_configs, J=J, h=h) # [2**D]
    log_z = log_pmf.logsumexp(dim=0)
    return log_z.item() if apply_log else log_z.exp().item()


def ising2d_get_pmf(L, J=1.0, h=0.0, beta=1.0, apply_log=False, device='cuda:0'):
    """
    Compute the (log) pmf of all configurations (in increasing order), shape [2**(L^2)]
    """
    all_configs = ising2d_get_all_configs(L, device)
    log_pmf = -beta * ising2d_ham(all_configs, J=J, h=h) # [2**D]
    return log_pmf.log_softmax(dim=0) if apply_log else log_pmf.softmax(dim=0)


def ising2d_visualize(S: torch.tensor, num_per_row: int = 8):
    """
    Visualize multiple Ising configurations in a grid.
    Args:
        S: 2D tensor of shape (N, D=L^2) representing N Ising configurations, values in {-1, 1}.
        num_per_row: Number of configurations to display per row.
    """
    plt.close('all')
    N, D = S.shape; L = int(D**0.5); 
    # assert L**2 == D, "The number of columns must be a perfect square."
    # assert torch.all((S == 1) | (S == -1)), "All entries of S must be either 1 or -1"
    num_rows = (N + num_per_row - 1) // num_per_row
    fig, axes = plt.subplots(num_rows, num_per_row, figsize=(num_per_row * 2, num_rows * 2))
    axes = axes.flatten()
    for i in range(len(axes)):
        if i < N:
            axes[i].imshow(S[i].reshape(L, L).cpu().numpy(), cmap='inferno', interpolation='nearest', vmin=-1, vmax=1)
        axes[i].axis('off')
    fig.tight_layout()
    return fig


##### sampling algorithms #####

def ising2d_mh(L, beta=0.5, J=1.0, h=0.0,
               batch_size=256, num_collect=20000, burn_in=10000, collect_every=1000, init=None):
    """
    Metropolis-Hastings algorithm to sample from the 2D Ising model's distribution.

    Parameters:
    - L: int, size of the lattice (L * L).
    - beta: float, inverse temperature (default=0.5).
    - J: float, interaction strength between neighboring spins (default=1.0).
    - h: float, external magnetic field (default=0.0).
    - batch_size: int, number of parallel configurations.
    - num_collect: int, number of times to collect.
    - burn_in: int, number of initial steps to discard (burn-in period).
    - collect_every: int, collect a sample every `collect_every` steps.
    - init: numpy.ndarray of shape (batch_size, L, L) or (batch_size, L * L), initial configuration.
            If None, random configurations are used.

    Returns:
    - samples: numpy.ndarray of shape (num_collect * batch_size, L, L), sampled configurations, values in {-1, 1}.
    """
    if init is not None:
        S = init.reshape(batch_size, L, L).astype(np.int16)
    else:
        S = np.random.choice([-1, 1], size=(batch_size, L, L)).astype(np.int16)

    samples = []
    batch_arange = np.arange(batch_size)

    pbar = tqdm(range(num_collect * collect_every + burn_in))
    for step in pbar:
        i, j = np.random.randint(0, L, size=(batch_size,)), np.random.randint(0, L, size=(batch_size,))
        dH = 2 * J * S[batch_arange, i, j] * (
            S[batch_arange, (i - 1) % L, j] + S[batch_arange, (i + 1) % L, j]
            + S[batch_arange, i, (j - 1) % L] + S[batch_arange, i, (j + 1) % L]
            ) + 2 * h * S[batch_arange, i, j]
        flip = np.random.rand(batch_size) < np.exp(-beta * dH)
        S[batch_arange[flip], i[flip], j[flip]] *= -1

        if step >= burn_in and (step - burn_in) % collect_every == 0:
            samples.append(np.copy(S))

    samples = np.array(samples).reshape(-1, L, L)
    np.random.shuffle(samples)
    return samples


def ising2d_wolff(L, beta=0.5, J=1.0, h=0.0,
                  batch_size=256, num_collect=20000, burn_in=10000, collect_every=1000, init=None):
    """
    (Metropolis-adjusted) Wolff cluster algorithm to sample from the 2D Ising model's distribution.

    Parameters:
    - L: int, size of the lattice (L * L).
    - beta: float, inverse temperature (default=0.5).
    - J: float, interaction strength between neighboring spins (default=1.0).
    - h: float, external magnetic field (default=0.0).
    - batch_size: int, number of parallel configurations.
    - num_collect: int, number of times to collect.
    - burn_in: int, number of initial steps to discard (burn-in period).
    - collect_every: int, collect a sample every `collect_every` steps.
    - init: numpy.ndarray of shape (batch_size, L, L) or (batch_size, L * L), initial configuration.
        If None, random configurations are used.

    Returns:
    - samples: numpy.ndarray of shape (num_collect * batch_size, L, L), sampled configurations, values in {-1, 1}.
    """
    if init is not None:
        S = init.reshape(batch_size, L, L).astype(np.int16)
    else:
        S = np.random.choice([-1, 1], size=(batch_size, L, L)).astype(np.int16)
    
    samples = []
    p_add = 1 - np.exp(-2 * beta * J)
    
    def grow_cluster(Sb, start_i, start_j):
        cluster = set([(start_i, start_j)])
        stack = [(start_i, start_j)]
        spin = Sb[start_i, start_j]
        while stack:
            ci, cj = stack.pop()
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = (ci + di) % L, (cj + dj) % L
                if (ni, nj) not in cluster and Sb[ni, nj] == spin:
                    if np.random.rand() < p_add:
                        cluster.add((ni, nj))
                        stack.append((ni, nj))
        return cluster
    
    pbar = tqdm(range(burn_in + num_collect * collect_every))
    for step in pbar:
        for b in range(batch_size):
            # pick a random position and grow a cluster
            i, j = np.random.randint(0, L, size=2)
            cluster = grow_cluster(S[b], i, j)

            if h == 0.0: # no external field, directly flip
                for ci, cj in cluster:
                    S[b, ci, cj] *= -1
            
            else: # with external field, use Metropolis-Hastings criterion
                # compute energy difference, only requires the external field part
                dH = 2 * h * sum(S[b, ci, cj] for ci, cj in cluster)
                if np.random.rand() < np.exp(-beta * dH):
                    for ci, cj in cluster:
                        S[b, ci, cj] *= -1

        if step >= burn_in and (step - burn_in) % collect_every == 0:
            samples.append(np.copy(S))

    samples = np.array(samples).reshape(-1, L, L)
    np.random.shuffle(samples)
    return samples


def ising2d_swendsen_wang(L, beta=0.5, J=1.0, h=0.0,
                          batch_size=256, num_collect=20000, burn_in=10000, collect_every=1000, init=None):
    # TODO: finish the Metropolis adjustment part
    """
    (Metropolis-adjusted) Swendsen-Wang algorithm to sample from the 2D Ising model's distribution.

    Parameters:
    - L: int, size of the lattice (L * L).
    - beta: float, inverse temperature (default=0.5).
    - J: float, interaction strength between neighboring spins (default=1.0).
    - h: float, external magnetic field (default=0.0).
    - batch_size: int, number of parallel configurations.
    - num_collect: int, number of times to collect.
    - burn_in: int, number of initial steps to discard (burn-in period).
    - collect_every: int, collect a sample every `collect_every` steps.
    - init: numpy.ndarray of shape (batch_size, L, L) or (batch_size, L * L), initial configuration.
            If None, random configurations are used.

    Returns:
    - samples: numpy.ndarray of shape (num_collect * batch_size, L, L), sampled configurations.
    """
    if init is not None:
        S = init.reshape(batch_size, L, L)
    else:
        S = np.random.choice([-1, 1], size=(batch_size, L, L))
    
    samples = []
    
    # Pre-compute bond probability
    p_bond = 1 - np.exp(-2 * beta * J)
    
    # Pre-allocate arrays for efficiency
    parent = np.zeros((batch_size, L**2), dtype=np.int32)
    rank = np.zeros((batch_size, L**2), dtype=np.int32)
    
    # Pre-compute neighbor indices for periodic boundary conditions
    indices = np.arange(L**2).reshape(L, L)
    right_neighbors = np.roll(indices, -1, axis=1).ravel()
    down_neighbors = np.roll(indices, -1, axis=0).ravel()
    
    def find(b, x):
        """Path compression find operation"""
        while parent[b, x] != x:
            parent[b, x] = parent[b, parent[b, x]]
            x = parent[b, x]
        return x
    
    def union(b, x, y):
        """Union by rank"""
        root_x = find(b, x)
        root_y = find(b, y)
        if root_x != root_y:
            if rank[b, root_x] < rank[b, root_y]:
                parent[b, root_x] = root_y
            else:
                parent[b, root_y] = root_x
                if rank[b, root_x] == rank[b, root_y]:
                    rank[b, root_x] += 1
    
    def process_configuration(b):
        # Step 1: Create bonds between aligned spins (fully vectorized)
        # Reshape for easier neighbor comparison
        S_flat = S[b].ravel()
        
        # Check horizontal bonds (right neighbors)
        aligned_h = S_flat == S_flat[right_neighbors]
        h_bonds = aligned_h & (np.random.random(L**2) < p_bond)
        
        # Check vertical bonds (down neighbors)
        aligned_v = S_flat == S_flat[down_neighbors]
        v_bonds = aligned_v & (np.random.random(L**2) < p_bond)
        
        # Step 2: Process bonds and identify clusters
        # Reset parent and rank arrays for this configuration
        parent[b] = np.arange(L**2, dtype=np.int32)
        rank[b].fill(0)
        
        # Process horizontal bonds
        h_indices = np.where(h_bonds)[0]
        for idx in h_indices:
            union(b, idx, right_neighbors[idx])
        
        # Process vertical bonds
        v_indices = np.where(v_bonds)[0]
        for idx in v_indices:
            union(b, idx, down_neighbors[idx])
        
        # Step 3: Identify clusters and flip them
        # Get unique cluster roots
        roots = np.array([find(b, i) for i in range(L**2)])
        unique_roots = np.unique(roots)
        
        # Generate flip decisions for each cluster
        flip_decisions = np.random.random(len(unique_roots)) < 0.5
        flip_map = dict(zip(unique_roots, flip_decisions))
        
        # Apply flips (vectorized)
        flip_mask = np.array([flip_map[root] for root in roots]).reshape(L, L)
        S[b] = np.where(flip_mask, -S[b], S[b])
    
    pbar = tqdm(range(burn_in + num_collect * collect_every))
    for step in pbar:
        # Process all configurations
        for b in range(batch_size):
            process_configuration(b)
        
        # Collect samples after burn-in period
        if step >= burn_in and (step - burn_in) % collect_every == 0:
            samples.append(np.copy(S))
    
    samples = np.array(samples).reshape(-1, L, L)
    np.random.shuffle(samples)
    return samples