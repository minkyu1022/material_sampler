import torch

# adds a constant to the position of a molecular system graph
def graph_add(graph, delta):
    graph["positions"] = graph["positions"] + delta
    return graph


# scale position vector of a molecular system graph
def graph_scale(graph, scalar):
    graph["positions"] = graph["positions"] * scalar
    return graph


def get_means(positions, batch_index):
    K = batch_index.max().item() + 1
    means = torch.zeros(K, positions.shape[1], dtype=positions.dtype).to(
        positions.device
    )
    means.index_reduce_(0, batch_index, positions, reduce="mean", include_self=False)
    return means


def subtract_com_batch(positions, batch_index):
    means = get_means(positions, batch_index)
    return positions - means[batch_index]


def is_zcom_positions(positions, batch_index, atol=1e-5):
    means = get_means(positions, batch_index)
    return torch.allclose(means, torch.zeros_like(means), atol=atol)


def is_zcom_graph(graph_state, atol=1e-5):
    return is_zcom_positions(
        graph_state["positions"],
        graph_state["batch"],
        atol=atol,
    )

@torch.no_grad()
def create_new_graph(graph_state, new_positions, subtract_com=False):
    new_graph_state = graph_state.clone()
    new_graph_state["positions"] = new_positions
    if subtract_com:
        new_graph_state["positions"] = subtract_com_batch(
            new_graph_state["positions"],
            new_graph_state["batch"],
        )
    return new_graph_state


# this assumes every system in the same molecule type. This is just for interfacing with standard torch models
def graph_to_vector(positions, n_particles, n_spatial_dim):
    n_systems = int(positions.shape[0] // n_particles)

    batch = positions.reshape(n_systems, n_spatial_dim * n_particles)
    return batch


def vector_to_graph(batch, n_particles, n_spatial_dim):
    n_systems = batch.shape[0]
    return batch.reshape(n_systems * n_particles, n_spatial_dim)


# def subtract_com_vector(samples, n_particles, spatial_dim):
#     shape = samples.shape
#     if isinstance(samples, torch.Tensor):
#         samples = samples.view(-1, n_particles, spatial_dim)
#         samples = samples - torch.mean(samples, dim=1, keepdim=True)
#         samples = samples.view(*shape)
#     else:
#         samples = samples.reshape(-1, n_particles, spatial_dim)
#         samples = samples - samples.mean(axis=1, keepdims=True)
#         samples = samples.reshape(*shape)
#     return samples

# same as subtract_com_vector
def remove_mean(samples, n_particles, spatial_dim):
    shape = samples.shape
    if isinstance(samples, torch.Tensor):
        samples = samples.view(-1, n_particles, spatial_dim)
        samples = samples - torch.mean(samples, dim=1, keepdim=True)
        samples = samples.view(*shape)
    else:
        samples = samples.reshape(-1, n_particles, spatial_dim)
        samples = samples - samples.mean(axis=1, keepdims=True)
        samples = samples.reshape(*shape)
    return samples

def is_freemean(samples, n_particles, spatial_dim, atol=1e-5):
    mean = samples.view(-1, n_particles, spatial_dim).mean(1)
    return torch.allclose(mean, torch.zeros_like(mean), atol=atol)
