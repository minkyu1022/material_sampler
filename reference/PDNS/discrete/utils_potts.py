"""
Utility functions for the 2D Potts models

Only MH sampling is implemented by numpy,
other functions are implemented by torch.

Please be aware of the input shape and range before use!
"""

from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


def potts2d_ham(S, J=1.0):
    r"""
    Compute the Hamiltonian for a batch of configurations in a 2D Potts model, with periodic boundary conditions.
    
    Parameters:
    - S: torch.tensor of shape (B, L * L):
        each element is range(q), representing spin configurations.
    - J: float, interaction strength between neighboring spins (default=1.0).
    Returns:
    - hamiltonians: torch.tensor of shape (B,) containing the Hamiltonian for each configuration.
        H(S) = -J * \sum_{i~j} \delta_{S_i, S_j} 
    (The p.m.f. is given by p(S) \propto e^{-\beta H(S)})
    """
    assert S.ndim == 2, "Input tensor must have shape (B, L * L)"
    S = S.view(S.size(0), int(S.shape[1]**.5), int(S.shape[1]**.5))
    equal_left = (S == torch.roll(S, shifts=1, dims=2)).int()
    equal_top = (S == torch.roll(S, shifts=1, dims=1)).int()
    return - float(J) * (equal_left + equal_top).sum(dim=(1, 2))


def potts2d_get_all_configs(L=4, q=3, device='cuda:0'):
    """
    Generate all possible Potts configurations for L x L lattice in increasing order
    Return: [q ** (L ** 2), L ** 2], values are in range(q)
    """
    arange = torch.arange(q ** (L ** 2), device=device)
    exponent = torch.arange(L ** 2 - 1, -1, -1, device=device)
    return (arange[:, None] // (q ** exponent)[None, :]) % q


def potts2d_emp_dist(samples, q=3):
    """
    samples: [B, L^2], elements in range(q)
    Output the empirical distribution of the samples as a probability vector of length q^{L^2}
    The configuations are sorted in increasing order as in potts2d_get_all_configs
    """
    q_powers = q ** torch.arange(samples.shape[1] - 1, -1, -1, device=samples.device)
    indices = (samples * q_powers).sum(dim=1)
    counts = torch.bincount(indices, minlength=q**samples.shape[1])
    return counts.float() / counts.sum()


def potts2d_get_par_func(L, J=1.0, beta=1.0, q=3, apply_log=False, device='cuda:0'):
    """
    Compute the (log) partition function of a 2D Potts model with periodic boundary conditions.
    """
    all_configs = potts2d_get_all_configs(L, q, device=device)
    log_pmf = -beta * potts2d_ham(all_configs, J)
    log_z = log_pmf.logsumexp(dim=0)
    return log_z.item() if apply_log else log_z.exp().item()


def potts2d_get_pmf(L, J=1.0, beta=1.0, q=3, apply_log=False, device='cuda:0'):
    """
    Compute the (log) pmf of all configurations (in increasing order), shape [q**(L^2)]
    """
    all_configs = potts2d_get_all_configs(L, q, device=device)
    log_pmf = -beta * potts2d_ham(all_configs, J)
    return log_pmf.log_softmax(dim=0) if apply_log else log_pmf.softmax(dim=0)





def potts2d_visualize(S: torch.tensor, num_per_row: int = 8, q: int = 3):
    """
    Visualize multiple Potts configurations in a grid.
    Args:
        S: 2D tensor of shape (N, D=L^2) representing N Potts configurations, values in range(q)
        num_per_row: Number of configurations to display per row.
    """
    N, D = S.shape; L = int(D**0.5); 
    assert L**2 == D, "The number of columns must be a perfect square."
    num_rows = (N + num_per_row - 1) // num_per_row
    fig, axes = plt.subplots(num_rows, num_per_row, figsize=(num_per_row * 2, num_rows * 2))
    axes = axes.flatten()
    
    for i in range(len(axes)):
        if i < N:
            axes[i].imshow(S[i].reshape(L, L).cpu().numpy(), cmap='inferno',
                           interpolation='nearest', vmin=0, vmax=q-1)
        axes[i].axis('off')
    fig.tight_layout()
    return fig


def potts2d_mag(S, q=3):
    """
    Compute the magnetization for a batch of configurations.

    Parameters:
    S: torch.tensor of shape (B, L * L) representing B configurations on an L x L lattice.
       Each element in S is in range(q).
    q: int, number of states in the Potts model.

    Returns:
    - a float: average magnetization over the batch.
    """
    B, D = S.shape; L = int(np.sqrt(D))
    assert L**2 == D, "The number of columns must be a perfect square."
    S = S.view(B, L, L)
    freq = (S[..., None] == torch.arange(q, device=S.device).view(1, 1, 1, -1)).float().mean(dim=0) # [L, L, q]
    mag = freq.max(dim=-1)[0] # [L, L]
    return (q * mag.mean().item() - 1) / (q - 1)


def potts2d_2pt_corr(S, r, q=3):
    """
    Calculate the 2-point correlation function of Potts model samples.
    Args:
        S (torch.Tensor): Potts model samples of shape (B, D=L^2)
        r (int): Distance between points to compute correlation
        q (int): Number of states (0 to q-1)
    Returns:
        torch.Tensor: average correlation, shape (B,)
    """
    B, D = S.shape; L = int(np.sqrt(D))
    assert L**2 == D, "The number of columns must be a perfect square."
    S = S.view(B, L, L)
    neighbors = [
        torch.roll(S, shifts=r, dims=1), torch.roll(S, shifts=-r, dims=1),
        torch.roll(S, shifts=r, dims=2), torch.roll(S, shifts=-r, dims=2),
    ]
    corr = sum((S == neighbor).int() for neighbor in neighbors) / 4
    return corr.mean() - 1/q


def potts2d_plot_2pt_corr(S, q=3):
    """
    Plot the 2-point correlation function for a batch of Potts configurations.

    Args:
        S: 2D tensor of shape (B, D=L^2) representing N Potts configurations, values in range(q).
    """
    B, D = S.shape; L = int(np.sqrt(D))
    assert L ** 2 == D, "The number of columns must be a perfect square."

    plt.close('all')
    fig = plt.figure()
    r = np.arange(-L//2, L//2 + 1)
    corr = [potts2d_2pt_corr(S, i, q).item() for i in r]
    plt.plot(r, corr, marker='o')
    plt.xlabel('Distance $r$')
    plt.xticks(r, [f'${i}$' for i in r])
    plt.ylim(-0.05, 1)
    plt.ylabel('2-point Correlation')
    return fig


#####

def potts2d_magnetization_all(S, q):
    # Convert to numpy if it's a torch tensor
    if isinstance(S, torch.Tensor):
        S = S.detach().cpu().numpy()
    
    # Reshape if needed
    if S.ndim == 2:
        B, L2 = S.shape
        L = int(np.sqrt(L2))
        S = S.reshape(B, L, L)
    else:
        B, L, L = S.shape
    
    # Compute the most frequent state for each configuration
    most_frequent = np.zeros(B)
    for b in range(B):
        # Count occurrences of each state
        counts = np.bincount(S[b].flatten(), minlength=q)
        # Get the most frequent state
        most_frequent[b] = np.argmax(counts)
    
    # Compute magnetization as the fraction of spins in the most frequent state
    magnetization = np.array([np.mean(S[b] == most_frequent[b]) for b in range(B)])
    return magnetization

def potts2d_magnetization(S, q, row=None, col=None):
    """
    Compute the magnetization of the 2D Potts model for a specific row or column.
    
    Args:
        S (torch.Tensor or numpy.ndarray): Potts model samples of shape (B, L, L) or (B, L*L)
        q (int): Number of states (0 to q-1)
        row (int, optional): Row index to compute magnetization for
        col (int, optional): Column index to compute magnetization for
    
    Returns:
        float: Magnetization value between 0 and 1 for the specified row or column
    """
    # Convert to numpy if it's a torch tensor
    if isinstance(S, torch.Tensor):
        S = S.detach().cpu().numpy()
    
    # Reshape if needed
    if S.ndim == 2:
        B, L2 = S.shape
        L = int(np.sqrt(L2))
        S = S.reshape(B, L, L)
    else:
        B, L, L = S.shape
    
    
    # Compute magnetization for specific row
    if row is not None:
        # Get the row data for all batches at once
        row_data = S[:, row, :]  # Shape: (B, L)
        # Count occurrences of each state for all batches at once
        counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=q), 1, row_data)  # Shape: (B, q)
        # Get most frequent state for each batch
        most_frequent = np.argmax(counts, axis=1)  # Shape: (B,)
        # Compute magnetization for all batches at once
        magnetization = (q * np.mean(row_data == most_frequent[:, None], axis=1) - 1) / (q - 1)
        return np.mean(magnetization)  # Average over batches
    
    # Compute magnetization for specific column
    elif col is not None:
        # Get the column data for all batches at once
        col_data = S[:, :, col]  # Shape: (B, L)
        # Count occurrences of each state for all batches at once
        counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=q), 1, col_data)  # Shape: (B, q)
        # Get most frequent state for each batch
        most_frequent = np.argmax(counts, axis=1)  # Shape: (B,)
        # Compute magnetization for all batches at once
        magnetization = (q * np.mean(col_data == most_frequent[:, None], axis=1) - 1) / (q - 1)
        return np.mean(magnetization)  # Average over batches
    
    else:
        raise ValueError("Either row or col must be specified")

def potts2d_magnetization_site(S, q):
    """
    Compute the magnetization for each individual site in the 2D Potts model.
    
    Args:
        S (torch.Tensor or numpy.ndarray): Potts model samples of shape (B, L, L) or (B, L*L)
        q (int): Number of states (0 to q-1)
    
    Returns:
        numpy.ndarray: LxL matrix where each entry (i,j) is the magnetization for that site
    """
    # Convert to numpy if it's a torch tensor
    if isinstance(S, torch.Tensor):
        S = S.detach().cpu().numpy()
    
    # Reshape if needed
    if S.ndim == 2:
        B, L2 = S.shape
        L = int(np.sqrt(L2))
        S = S.reshape(B, L, L)
    else:
        B, L, L = S.shape
    
    # Initialize magnetization matrix
    magnetization = np.zeros((L, L))
    
    # For each site, compute its magnetization
    for i in range(L):
        for j in range(L):
            # Get the state at this site for all batches
            site_states = S[:, i, j]  # Shape: (B,)
            # Count occurrences of each state
            counts = np.bincount(site_states, minlength=q)
            # Get the most frequent state
            most_frequent = np.argmax(counts)
            # Compute magnetization for this site
            magnetization[i, j] = (q * np.mean(site_states == most_frequent) - 1) / (q - 1)
    
    return magnetization

def potts2d_magnetization_ij(S, q):
    """
    Compute the magnetization for all sites in the 2D Potts model.
    
    Args:
        S (torch.Tensor or numpy.ndarray): Potts model samples of shape (B, L, L) or (B, L*L)
        q (int): Number of states (0 to q-1)
    
    Returns:
        numpy.ndarray: LxL matrix where each entry (i,j) is the magnetization at that site
    """
    # Convert to numpy if it's a torch tensor
    if isinstance(S, torch.Tensor):
        S = S.detach().cpu().numpy()
    
    # Reshape if needed
    if S.ndim == 2:
        B, L2 = S.shape
        L = int(np.sqrt(L2))
        S = S.reshape(B, L, L)
    else:
        B, L, L = S.shape
    
    # Reshape S to (B, L*L) for easier processing
    S_flat = S.reshape(B, L*L)  # Shape: (B, L*L)
    
    # Initialize magnetization array
    magnetization = np.zeros(L*L)
    
    # Process each site
    for i in range(L*L):
        # Get states for this site across all batches
        site_states = S_flat[:, i]  # Shape: (B,)
        # Count occurrences of each state
        counts = np.bincount(site_states, minlength=q)
        # Get most frequent state
        most_frequent = np.argmax(counts)
        # Compute magnetization for this site
        magnetization[i] = (q * np.mean(site_states == most_frequent) - 1) / (q - 1)
    
    # Reshape to (L, L) for the final result
    magnetization = magnetization.reshape(L, L)  # Shape: (L, L)
    
    return magnetization


def potts2d_mh(L, beta=.5, J=1.0, h=0.0, q=3, batch_size=256, num_collect=20000, 
               burn_in=10000, collect_every=1000, init=None):
    """
    Metropolis-Hastings algorithm to sample from the 2D Potts model's distribution.

    Parameters:
    - L: int, size of the lattice (L * L).
    - beta: float, inverse temperature.
    - J: float, coupling constant
    - h: float, external field. The current version only supports h = 0.
    - q: int, number of states (0 to q-1)
    - batch_size: int, number of parallel configurations.
    - num_collect: int, number of times to collect.
    - burn_in: int, number of initial steps to discard (burn-in period).
    - collect_every: int, collect a sample every `collect_every` steps.
    - init: numpy.ndarray of shape (B, L, L) or (B, L * L), initial configuration.
            If None, random configurations are used.
    Returns:
    - samples: numpy.ndarray of shape (num_collect * B, L * L), sampled configurations.
    """
    if init is None:
        S = np.random.randint(0, q, size=(batch_size, L, L))
    else:
        S = init.reshape(batch_size, L, L) if init.ndim == 2 else init
    
    samples = []
    total_steps = burn_in + num_collect * collect_every
    arange_B = np.arange(batch_size)
    
    for step in tqdm(range(total_steps)):
        i = np.random.randint(0, L, size=batch_size); j = np.random.randint(0, L, size=batch_size)
        
        current_spins = S[arange_B, i, j]
        new_spins = np.random.randint(0, q, size=batch_size)
        while True:
            mask = new_spins == current_spins
            if not np.any(mask):
                break # Ensure new spin is different from current spin
            new_spins[mask] = np.random.randint(0, q, size=np.sum(mask))
        
        left = S[arange_B, i, (j-1)%L]; right = S[arange_B, i, (j+1)%L]
        up = S[arange_B, (i-1)%L, j]; down = S[arange_B, (i+1)%L, j]
        
        H_old = -J * ((current_spins == left) + (current_spins == right) + 
                      (current_spins == up) +  (current_spins == down))
        H_new = -J * ((new_spins == left) + (new_spins == right) + 
                      (new_spins == up) + (new_spins == down))
        accept = np.random.random(size=batch_size) < np.exp(-beta * (H_new - H_old))
        S[arange_B[accept], i[accept], j[accept]] = new_spins[accept]
        
        if step >= burn_in and (step - burn_in) % collect_every == 0:
            samples.append(np.copy(S))
    return np.array(samples).reshape(-1, L*L)


def potts2d_glauber(L, beta=.5, J=1.0, h=0.0, q=3, batch_size=256, num_collect=20000, 
                    burn_in=10000, collect_every=1000, init=None):
    """
    Glauber dynamics algorithm to sample from the 2D Potts model's distribution.
    Optimized version with vectorized operations.
    """
    # Initialize the lattice
    if init is None:
        S = np.random.randint(0, q, size=(batch_size, L, L))
    else:
        S = init.reshape(batch_size, L, L) if init.ndim == 2 else init
    
    samples = []
    total_steps = burn_in + num_collect * collect_every
    batch_arange = np.arange(batch_size)
    
    # Pre-allocate arrays
    local_fields = np.zeros((batch_size, q))
    exp_fields = np.zeros((batch_size, q))
    
    betaJ = -beta * (-J)
    
    for step in tqdm(range(total_steps)):
        # Randomly select sites to update
        i = np.random.randint(0, L, size=batch_size)
        j = np.random.randint(0, L, size=batch_size)
        
        # Get neighbors with periodic boundary conditions
        left = S[batch_arange, i, (j-1)%L]
        right = S[batch_arange, i, (j+1)%L]
        up = S[batch_arange, (i-1)%L, j]
        down = S[batch_arange, (i+1)%L, j]
        
        # Vectorized calculation of local fields for all states at once
        # Create a (B, q) array where each row is [0,1,...,q-1]
        states = np.arange(q)[None, :].repeat(batch_size, axis=0)
        
        # Calculate matching neighbors for all states at once
        # Shape: (B, q) - each element is number of matching neighbors for that state
        matches = ((states == left[:, None]).astype(int) + 
                  (states == right[:, None]).astype(int) + 
                  (states == up[:, None]).astype(int) + 
                  (states == down[:, None]).astype(int))
        
        # Calculate local fields
        local_fields = betaJ * matches
        
        # Calculate probabilities using softmax (vectorized)
        exp_fields = np.exp(local_fields - np.max(local_fields, axis=1, keepdims=True))
        probs = exp_fields / np.sum(exp_fields, axis=1, keepdims=True)
        
        # Sample new states according to probabilities
        # Vectorized sampling using cumsum trick
        cumsum = np.cumsum(probs, axis=1)
        r = np.random.random(size=batch_size)[:, None]
        new_spins = np.argmax(cumsum > r, axis=1)
        
        # Update spins
        S[batch_arange, i, j] = new_spins
        
        # Collect samples after burn-in
        if step >= burn_in and (step - burn_in) % collect_every == 0:
            samples.append(S.reshape(batch_size, L*L).copy())
    
    return np.concatenate(samples, axis=0)


def potts2d_swendsen_wang(L, beta=.5, J=1.0, h=0.0, q=3, batch_size=256, num_collect=20000, 
                         burn_in=10000, collect_every=1000, init=None):
    """
    Swendsen-Wang algorithm to sample from the 2D Potts model's distribution.
    
    Parameters:
    - L: int, size of the lattice (L * L)
    - beta: float, inverse temperature
    - J: float, coupling constant
    - h: float, external magnetic field. The current version only supports h = 0.
    - q: int, number of states (0 to q-1)
    - B: int, number of parallel configurations
    - num_collect: int, number of times to collect
    - burn_in: int, number of initial steps to discard
    - collect_every: int, collect a sample every `collect_every` steps
    - init: numpy.ndarray of shape (B, L, L) or (B, L * L), initial configuration
    
    Returns:
    - samples: numpy.ndarray of shape (num_collect * B, L * L), sampled configurations
    """
    # Initialize the lattice
    if init is None:
        S = np.random.randint(0, q, size=(batch_size, L, L))
    else:
        S = init.reshape(batch_size, L, L) if init.ndim == 2 else init
    
    samples = []
    total_steps = burn_in + num_collect * collect_every
    
    # Pre-compute bond probability
    p = 1 - np.exp(-beta * J)
    
    for step in tqdm(range(total_steps)):
        # For each configuration in the batch
        for b in range(batch_size):
            # Step 1: Identify bonds between same-state neighbors
            # Create arrays for horizontal and vertical bonds
            h_bonds = np.zeros((L, L), dtype=bool)  # horizontal bonds
            v_bonds = np.zeros((L, L), dtype=bool)  # vertical bonds
            
            # Check horizontal bonds
            h_bonds[:, :-1] = (S[b, :, :-1] == S[b, :, 1:])
            h_bonds[:, -1] = (S[b, :, -1] == S[b, :, 0])  # periodic BC
            
            # Check vertical bonds
            v_bonds[:-1, :] = (S[b, :-1, :] == S[b, 1:, :])
            v_bonds[-1, :] = (S[b, -1, :] == S[b, 0, :])  # periodic BC
            
            # Step 2: Activate bonds with probability p
            h_bonds = h_bonds & (np.random.random((L, L)) < p)
            v_bonds = v_bonds & (np.random.random((L, L)) < p)
            
            # Step 3: Identify clusters using Union-Find
            # Initialize parent array for Union-Find
            parent = np.arange(L * L).reshape(L, L)
            rank = np.zeros((L, L), dtype=int)
            
            def find(x, y):
                if parent[x, y] != x * L + y:
                    px, py = parent[x, y] // L, parent[x, y] % L
                    parent[x, y] = find(px, py)
                return parent[x, y]
            
            def union(x1, y1, x2, y2):
                root1 = find(x1, y1)
                root2 = find(x2, y2)
                if root1 != root2:
                    r1, c1 = root1 // L, root1 % L
                    r2, c2 = root2 // L, root2 % L
                    if rank[r1, c1] < rank[r2, c2]:
                        parent[r1, c1] = root2
                    else:
                        parent[r2, c2] = root1
                        if rank[r1, c1] == rank[r2, c2]:
                            rank[r1, c1] += 1
            
            # Process horizontal bonds
            for i in range(L):
                for j in range(L):
                    if h_bonds[i, j]:
                        union(i, j, i, (j + 1) % L)
            
            # Process vertical bonds
            for i in range(L):
                for j in range(L):
                    if v_bonds[i, j]:
                        union(i, j, (i + 1) % L, j)
            
            # Step 4: Identify clusters
            clusters = {}
            for i in range(L):
                for j in range(L):
                    root = find(i, j)
                    if root not in clusters:
                        clusters[root] = []
                    clusters[root].append((i, j))
            
            # Step 5: Flip clusters
            for cluster in clusters.values():
                # Randomly choose new state for the cluster
                new_state = np.random.randint(0, q)
                # Update all spins in the cluster
                for i, j in cluster:
                    S[b, i, j] = new_state
        
        # Collect samples after burn-in
        if step >= burn_in and (step - burn_in) % collect_every == 0:
            samples.append(S.reshape(batch_size, L*L).copy())
    
    return np.concatenate(samples, axis=0)
