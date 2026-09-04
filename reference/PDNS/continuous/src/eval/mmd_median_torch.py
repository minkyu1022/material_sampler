import torch
import torch.nn.functional as F

def torch_distances(X, Y, l, max_samples=None, matrix=False):
    X, Y = X[:max_samples], Y[:max_samples]
    if l == "l1":
        pairwise_dist = torch.cdist(X, Y, p=1)
    elif l == "l2":
        pairwise_dist = torch.cdist(X, Y, p=2)
    else:
        raise ValueError("Value of 'l' must be either 'l1' or 'l2'.")
    
    return pairwise_dist if matrix else pairwise_dist[torch.triu_indices(pairwise_dist.shape[0], pairwise_dist.shape[1])]

def kernel_matrix(pairwise_matrix, l, kernel, bandwidth, rq_kernel_exponent=0.5):
    d = pairwise_matrix / bandwidth
    if kernel == "gaussian" and l == "l2":
        return torch.exp(-0.5 * (d ** 2))
    elif kernel == "laplace" and l == "l1":
        return torch.exp(-d * torch.sqrt(torch.tensor(2.0)))
    elif kernel == "rq" and l == "l2":
        return (1 + d ** 2 / (2 * rq_kernel_exponent)) ** (-rq_kernel_exponent)
    elif kernel == "imq" and l == "l2":
        return (1 + d ** 2) ** (-0.5)
    elif kernel.startswith("matern"):
        order = float(kernel.split("_")[1])
        if order == 0.5:
            return torch.exp(-d)
        elif order == 1.5:
            return (1 + torch.sqrt(torch.tensor(3.0)) * d) * torch.exp(-torch.sqrt(torch.tensor(3.0)) * d)
        elif order == 2.5:
            return (1 + torch.sqrt(torch.tensor(5.0)) * d + 5/3 * d ** 2) * torch.exp(-torch.sqrt(torch.tensor(5.0)) * d)
        elif order == 3.5:
            return (1 + torch.sqrt(torch.tensor(7.0)) * d + 2 * 7/5 * d ** 2 + 7 * torch.sqrt(torch.tensor(7.0)) / 15 * d ** 3) * torch.exp(-torch.sqrt(torch.tensor(7.0)) * d)
        elif order == 4.5:
            return (1 + 3 * d + 3 * (6 ** 2) / 28 * d ** 2 + (6 ** 3) / 84 * d ** 3 + (6 ** 4) / 1680 * d ** 4) * torch.exp(-3 * d)
    raise ValueError('The values of "l" and "kernel" are not valid.')

def median_mmd(X, Y):
    m, n = X.shape[0], Y.shape[0]
    assert m >= 2 and n >= 2
    kernel, l = 'gaussian', "l2"
    
    Z = torch.cat((X, Y), dim=0)
    pairwise_matrix = torch_distances(Z, Z, l, matrix=True)
    indices = torch.triu_indices(pairwise_matrix.shape[0], pairwise_matrix.shape[1])
    distances = pairwise_matrix[indices[0], indices[1]]
    bandwidth = torch.median(distances)
    
    d_XY, d_XX, d_YY = torch_distances(X, Y, l, matrix=True), torch_distances(X, X, l, matrix=True), torch_distances(Y, Y, l, matrix=True)
    K_XY, K_XX, K_YY = kernel_matrix(d_XY, l, kernel, bandwidth), kernel_matrix(d_XX, l, kernel, bandwidth), kernel_matrix(d_YY, l, kernel, bandwidth)
    
    mmd = (K_XX.sum() / (m * (m - 1))) + (K_YY.sum() / (n * (n - 1))) - 2 * K_XY.mean()
    return torch.sqrt(torch.clamp(mmd, min=1e-20))

def compute_bandwidths(X, Y, l, number_bandwidths, only_median=False):
    Z = torch.cat((X, Y), dim=0)
    distances = torch_distances(Z, Z, l, matrix=False)
    median = torch.median(distances)
    if only_median:
        return median
    distances = distances + (distances == 0) * median
    sorted_distances = torch.sort(distances).values
    lambda_min = sorted_distances[int(len(sorted_distances) * 0.05)] / 2
    lambda_max = sorted_distances[int(len(sorted_distances) * 0.95)] * 2
    return torch.linspace(lambda_min, lambda_max, number_bandwidths)
