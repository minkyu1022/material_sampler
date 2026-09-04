import math
from typing import Union

import numpy as np
import torch

from .mmd import linear_mmd2, mix_rbf_mmd2, poly_mmd2, _mmd2
from .mmd_median_torch import median_mmd
from .optimal_transport import wasserstein
from geomloss import SamplesLoss

def compute_distances(pred, true):
    """Computes distances between vectors."""
    mse = torch.nn.functional.mse_loss(pred, true).item()
    me = math.sqrt(mse)
    mae = torch.mean(torch.abs(pred - true)).item()
    return mse, me, mae


def compute_distribution_distances(
    pred: torch.Tensor, true: Union[torch.Tensor, list], local: torch.Tensor, energy_function
):
    """computes distances between distributions.
    pred: [batch, times, dims] tensor
    true: [batch, times, dims] tensor or list[batch[i], dims] of length times

    This handles jagged times as a list of tensors.
    """

    names = [
        # "1-Wasserstein",
        "2-Wasserstein",
        "local_2-Wasserstein",
        "Sinkhorn",
        "local_Sinkhorn",
        "Median_MMD",
        "local_Median_MMD"
        # "Linear_MMD",
        # "Poly_MMD",
        # "RBF_MMD",
        # "Eq-W1",
        # "Eq-W2",
    ]

    # w1 = wasserstein(pred, true, power=1)
    w2 = wasserstein(pred, true, power=2)
    local_w2 = wasserstein(pred, local, power=2)

    # geomloss uses 1/2||x-y||^2 cost; multiply by 2 to match SCLD convention.
    # truncate=None + scaling=0.99 prevents early termination on wide supports (e.g. GMM40 [-40,40]^50) with tiny blur.
    sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2, blur=1e-3, truncate=None, scaling=0.99)
    sinkhorn = 2.0 * sinkhorn_loss(true, pred).detach().cpu().numpy()
    local_sinkhorn = 2.0 * sinkhorn_loss(local, pred).detach().cpu().numpy()

    mmd_median = median_mmd(true, pred).item()
    local_mmd_median = median_mmd(local, pred).item()
    # mmd_linear = linear_mmd2(a, b).item()
    # mmd_poly = poly_mmd2(a, b, d=2, alpha=1.0, c=2.0).item()
    # mmd_rbf = mix_rbf_mmd2(a, b, sigma_list=[0.01, 0.1, 1, 10, 100]).item()

    # dists = [w1,
    #         w2,
    #         sinkhorn,
    #         mmd_median,
    #         mmd_linear,
    #         mmd_poly,
    #         mmd_rbf,
    #         ]

    dist = [w2,
            local_w2,
            sinkhorn,
            local_sinkhorn,
            mmd_median,
            local_mmd_median]
    
    return {name: val for name, val in zip(names, dist)}

def compute_distribution_distances2(
    pred: torch.Tensor, true: Union[torch.Tensor, list], local: torch.Tensor, energy_function
):
    """computes distances between distributions.
    pred: [batch, times, dims] tensor
    true: [batch, times, dims] tensor or list[batch[i], dims] of length times

    This handles jagged times as a list of tensors.
    """

    names = [
        # "1-Wasserstein",
        "2-Wasserstein",
        "local_2-Wasserstein",
        "Eq-W1",
        "Eq-W2",
        "local_Eq-W1",        
        "local_Eq-W2"
    ]

    # w1 = wasserstein(a, b, power=1)
    w2 = wasserstein(pred, true, power=2)
    local_w2 = wasserstein(pred, local, power=2)

    M = dist_point_clouds(
        pred.reshape(
            -1, energy_function.n_particles, energy_function.n_spatial_dim
        ).cpu(),
        true.reshape(
            -1, energy_function.n_particles, energy_function.n_spatial_dim
        ).cpu(),
    )
    eq_w1, eq_w2 = wassersteins_from_dist(M)

    M = dist_point_clouds(
        pred.reshape(
            -1, energy_function.n_particles, energy_function.n_spatial_dim
        ).cpu(),
        local.reshape(
            -1, energy_function.n_particles, energy_function.n_spatial_dim
        ).cpu(),
    )
    local_eq_w1, local_eq_w2 = wassersteins_from_dist(M)

    # dists = [w1,
    #         w2,
    #         eq_w1,
    #         eq_w2]

    dists = [w2, local_w2,
            eq_w1, eq_w2,
            local_eq_w1, local_eq_w2]

    return {name: val for name, val in zip(names, dists)}


def compute_w_distances(
    pred: torch.Tensor, true: torch.Tensor
):
    """computes distances between distributions.
    pred: [batch, times, dims] tensor
    true: [batch, times, dims] tensor or list[batch[i], dims] of length times

    This handles jagged times as a list of tensors.
    """
    w1 = wasserstein(pred, true, power=1)
    w2 = wasserstein(pred, true, power=2)
    return {"1-Wasserstein":w1, "2-Wasserstein":w2}


import ot as pot
from scipy.optimize import linear_sum_assignment


def find_rigid_alignment(A, B):
    """
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    2-D or 3-D registration with known correspondences.
    Registration occurs in the zero centered coordinate system, and then
    must be transported back.
        Args:
        -    A: Torch tensor of shape (N,D) -- Point Cloud to Align (source)
        -    B: Torch tensor of shape (N,D) -- Reference Point Cloud (target)
        Returns:
        -    R: optimal rotation
        -    t: optimal translation
    Test on rotation + translation and on rotation + translation + reflection
        >>> A = torch.tensor([[1., 1.], [2., 2.], [1.5, 3.]], dtype=torch.float)
        >>> R0 = torch.tensor([[np.cos(60), -np.sin(60)], [np.sin(60), np.cos(60)]], dtype=torch.float)
        >>> B = (R0.mm(A.T)).T
        >>> t0 = torch.tensor([3., 3.])
        >>> B += t0
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
        >>> B *= torch.tensor([-1., 1.])
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
    """
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.T.mm(B_c)
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = V.mm(U.T)
    # Translation vector
    t = b_mean[None, :] - R.mm(a_mean[None, :].T).T
    t = t.T
    return R, t.squeeze()


def ot(x0, x1):
    dists = torch.cdist(x0, x1)
    _, col_ind = linear_sum_assignment(dists)
    x1 = x1[col_ind]
    return x1


def dist_point_clouds(x0, x1):
    M = []
    for i in range(len(x0)):
        reordered = []
        for j in range(len(x1)):
            x1_reordered = ot(x0[i], x1[j])
            reordered.append(x1_reordered)
        reordered = torch.stack(reordered)
        R, t = torch.vmap(find_rigid_alignment)(
            x0[i][None].repeat(len(x1), 1, 1), reordered
        )
        superimposed = torch.matmul(reordered, R)
        M.append(torch.cdist(x0[i].reshape(1, -1), superimposed.reshape(len(x1), -1)))
    M = torch.stack(M).squeeze()
    return M


def eot(x0, x1):
    M = dist_point_clouds(x0, x1)
    return pot.emd2(
        M=M, a=torch.ones(len(x0)) / len(x0), b=torch.ones(len(x1)) / len(x1)
    )

def wassersteins_from_dist(M):
    n = M.shape[0]
    W1 =  pot.emd2(
        M=M, a=torch.ones(n) / n, b=torch.ones(n) / n
    )
    W2 = pot.emd2(
        M=M**2, a=torch.ones(n) / n, b=torch.ones(n) / n
    )**0.5
    return W1, W2


def _mix_rbf_kernel(x0, x1, sigma_list):
    m = x0.shape[0]
    Z = torch.cat((x0, x1), 0)
    exponent = dist_point_clouds(Z, Z) ** 2

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        K += torch.exp(-gamma * exponent)

    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)


def mix_rbf_mmd2_point_cloud(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)