"""Small, runnable reproduction of JANUS' discrete Ising experiment."""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import product

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def ising_energy(
    spins: np.ndarray | Tensor, coupling: float = 1.0, delta_mu: float | Tensor = 0.0
) -> np.ndarray | Tensor:
    """Periodic square-lattice energy, up to the irrelevant ``-delta_mu*N/2`` constant."""
    if torch.is_tensor(spins):
        bonds = spins * (torch.roll(spins, -1, -1) + torch.roll(spins, -1, -2))
        field = torch.as_tensor(delta_mu, device=spins.device, dtype=spins.dtype)
        while field.ndim < spins.ndim - 2:
            field = field.unsqueeze(-1)
        return -coupling * bonds.sum((-2, -1)) - 0.5 * field * spins.sum((-2, -1))
    spins = np.asarray(spins)
    bonds = spins * (np.roll(spins, -1, -1) + np.roll(spins, -1, -2))
    return -coupling * bonds.sum(axis=(-2, -1)) - 0.5 * np.asarray(delta_mu) * spins.sum(
        axis=(-2, -1)
    )


energy = ising_energy


def heat_bath_prob(
    spins: np.ndarray | Tensor,
    temperature: float | np.ndarray | Tensor,
    delta_mu: float | np.ndarray | Tensor = 0.0,
    coupling: float = 1.0,
) -> np.ndarray | Tensor:
    """Return the exact single-site conditional probability ``P(spin=+1 | rest)``."""
    neighbours = sum(
        (torch.roll(spins, shift, dim) if torch.is_tensor(spins) else np.roll(spins, shift, dim))
        for dim in (-2, -1)
        for shift in (-1, 1)
    )
    if torch.is_tensor(spins):
        t = torch.as_tensor(temperature, device=spins.device, dtype=spins.dtype)
        dmu = torch.as_tensor(delta_mu, device=spins.device, dtype=spins.dtype)
        while t.ndim < spins.ndim:
            t = t.unsqueeze(-1)
        while dmu.ndim < spins.ndim:
            dmu = dmu.unsqueeze(-1)
        return torch.sigmoid((2.0 * coupling * neighbours + dmu) / t)
    t = np.asarray(temperature)
    dmu = np.asarray(delta_mu)
    while t.ndim < spins.ndim:
        t = np.expand_dims(t, -1)
    while dmu.ndim < spins.ndim:
        dmu = np.expand_dims(dmu, -1)
    return 1.0 / (1.0 + np.exp(-(2.0 * coupling * neighbours + dmu) / t))


heat_bath_labels = heat_bath_prob


def wolff_step(
    spins: np.ndarray,
    temperature: float,
    delta_mu: float = 0.0,
    coupling: float = 1.0,
    rng: np.random.Generator | None = None,
) -> int:
    """Perform one ghost-spin Wolff update in-place and return the cluster size."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if spins.ndim != 2 or spins.shape[0] != spins.shape[1]:
        raise ValueError("spins must be a square 2-D array")
    rng = rng or np.random.default_rng()
    length = spins.shape[0]
    seed = tuple(rng.integers(length, size=2))
    seed_spin = spins[seed]
    cluster = {seed}
    stack = [seed]
    p_bond = 1.0 - math.exp(-2.0 * coupling / temperature)
    ghost_spin = 1 if delta_mu >= 0 else -1
    p_ghost = 1.0 - math.exp(-abs(delta_mu) / temperature)  # 2|h|, h=delta_mu/2
    touches_ghost = False

    while stack:
        i, j = stack.pop()
        if seed_spin == ghost_spin and rng.random() < p_ghost:
            touches_ghost = True
        for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            site = (ni % length, nj % length)
            if site not in cluster and spins[site] == seed_spin and rng.random() < p_bond:
                cluster.add(site)
                stack.append(site)
    if not touches_ghost:
        rows, cols = zip(*cluster)
        spins[rows, cols] *= -1
    return len(cluster)


def ghost_wolff_samples(
    length: int,
    temperature: float,
    delta_mu: float = 0.0,
    *,
    num_samples: int = 1000,
    burn_in: int = 200,
    chains: int = 1,
    steps_per_sample: int = 1,
    coupling: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate reference configurations with independent ghost-Wolff chains."""
    if min(length, num_samples, chains, steps_per_sample) < 1 or burn_in < 0:
        raise ValueError("sizes must be positive and burn_in non-negative")
    rng = np.random.default_rng(seed)
    states = rng.choice(np.array([-1, 1], dtype=np.int8), (chains, length, length))
    out = np.empty((num_samples, chains, length, length), dtype=np.int8)
    for chain in range(chains):
        for _ in range(burn_in):
            wolff_step(states[chain], temperature, delta_mu, coupling, rng)
    for sample in range(num_samples):
        for chain in range(chains):
            for _ in range(steps_per_sample):
                wolff_step(states[chain], temperature, delta_mu, coupling, rng)
        out[sample] = states
    return out[:, 0] if chains == 1 else out


reference_samples = ghost_wolff_samples


def observables(
    spins: np.ndarray | Tensor,
    coupling: float = 1.0,
    delta_mu: float | Tensor = 0.0,
) -> dict[str, float]:
    """Compute the Figure-2 observables and mean energy per site."""
    if torch.is_tensor(spins):
        x = spins.detach().float()
        magnetization = x.mean((-2, -1))
        e = ising_energy(x, coupling, delta_mu) / (x.shape[-1] * x.shape[-2])
        return {
            "up_fraction": float(((x + 1) * 0.5).mean()),
            "magnetization": float(magnetization.mean()),
            "abs_magnetization": float(magnetization.abs().mean()),
            "energy_per_site": float(e.mean()),
        }
    x = np.asarray(spins, dtype=float)
    magnetization = x.mean(axis=(-2, -1))
    e = ising_energy(x, coupling, delta_mu) / (x.shape[-1] * x.shape[-2])
    return {
        "up_fraction": float(((x + 1) * 0.5).mean()),
        "magnetization": float(magnetization.mean()),
        "abs_magnetization": float(np.abs(magnetization).mean()),
        "energy_per_site": float(np.mean(e)),
    }


metrics = observables


def exact_ising_observables(
    length: int,
    temperature: float,
    delta_mu: float = 0.0,
    coupling: float = 1.0,
) -> dict[str, float]:
    """Enumerate the exact finite-lattice ensemble (intended for ``length <= 4``)."""
    if length < 1 or length > 4:
        raise ValueError("exact enumeration is limited to lattices of length 1 through 4")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    states = np.fromiter(
        (spin for state in product((-1, 1), repeat=length * length) for spin in state),
        dtype=np.int8,
    ).reshape(-1, length, length)
    log_weights = -ising_energy(states, coupling, delta_mu) / temperature
    weights = np.exp(log_weights - log_weights.max())
    weights /= weights.sum()
    magnetization = states.mean(axis=(-2, -1))
    energy_per_site = ising_energy(states, coupling, delta_mu) / (length * length)
    return {
        "up_fraction": float(np.sum(weights * (magnetization + 1.0) * 0.5)),
        "magnetization": float(np.sum(weights * magnetization)),
        "abs_magnetization": float(np.sum(weights * np.abs(magnetization))),
        "energy_per_site": float(np.sum(weights * energy_per_site)),
    }


class MaskedIsingConv(nn.Module):
    """Periodic fully-convolutional JANUS species head; zero-init gives fair coins."""

    def __init__(self, width: int = 64, depth: int = 4):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be positive")
        self.input = nn.Conv2d(6, width, 3, padding=1, padding_mode="circular")
        self.blocks = nn.ModuleList(
            nn.Conv2d(2 * width + 3, width, 3, padding=1, padding_mode="circular")
            for _ in range(depth - 1)
        )
        self.output = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _batch(value: float | Tensor, state: Tensor) -> Tensor:
        value = torch.as_tensor(value, device=state.device, dtype=torch.float32)
        if value.ndim == 0:
            value = value.expand(state.shape[0])
        return value.reshape(state.shape[0], 1, 1, 1).expand(-1, 1, *state.shape[-2:])

    def forward(
        self,
        state: Tensor,
        time: float | Tensor,
        temperature: float | Tensor,
        delta_mu: float | Tensor = 0.0,
    ) -> Tensor:
        if state.ndim == 2:
            state = state[None]
        state = state.float()
        cond = torch.cat(
            (
                self._batch(time, state),
                1.0 / self._batch(temperature, state),
                self._batch(delta_mu, state),
            ),
            1,
        )
        one_hot = torch.stack((state.eq(-1), state.eq(0), state.eq(1)), 1).float()
        hidden = F.silu(self.input(torch.cat((one_hot, cond), 1)))
        for conv in self.blocks:
            mean = hidden.mean((-2, -1), keepdim=True).expand_as(hidden)
            hidden = hidden + F.silu(conv(torch.cat((hidden, mean, cond), 1)))
        return self.output(hidden).squeeze(1)


JANUSIsing = MaskedIsingConv


def masked_soft_ce(
    model: nn.Module,
    terminals: Tensor,
    temperature: float | Tensor,
    delta_mu: float | Tensor = 0.0,
    coupling: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """JANUS soft-label objective on randomly re-masked terminal configurations."""
    terminals = terminals.float()
    batch = terminals.shape[0]
    time = torch.rand(batch, device=terminals.device, generator=generator)
    masked = (
        torch.rand(terminals.shape, device=terminals.device, generator=generator)
        > time[:, None, None]
    )
    state = terminals.masked_fill(masked, 0)
    targets = heat_bath_prob(terminals, temperature, delta_mu, coupling)
    loss = F.binary_cross_entropy_with_logits(
        model(state, time, temperature, delta_mu), targets, reduction="none"
    )
    return (loss * masked).sum() / masked.sum().clamp_min(1)


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    terminals: Tensor,
    temperature: float | Tensor,
    delta_mu: float | Tensor = 0.0,
    coupling: float = 1.0,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss = masked_soft_ce(model, terminals, temperature, delta_mu, coupling)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def sample_janus(
    model: nn.Module,
    batch_size: int,
    length: int,
    temperature: float | Tensor,
    delta_mu: float | Tensor = 0.0,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    return_log_prob: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Any-order autoregressive unmasking; each lattice site is revealed exactly once."""
    device = device or next(model.parameters()).device
    state = torch.zeros((batch_size, length, length), device=device)
    order = torch.rand((batch_size, length * length), device=device, generator=generator).argsort(1)
    log_prob = torch.zeros(batch_size, device=device)
    rows = torch.arange(batch_size, device=device)
    for step in range(length * length):
        logits = model(state, step / (length * length), temperature, delta_mu).flatten(1)
        site = order[:, step]
        chosen_logits = logits[rows, site]
        plus = torch.bernoulli(torch.sigmoid(chosen_logits), generator=generator)
        state.view(batch_size, -1)[rows, site] = plus.mul(2).sub(1)
        log_prob += -F.binary_cross_entropy_with_logits(chosen_logits, plus, reduction="none")
    return (state, log_prob) if return_log_prob else state


sample = sample_janus


def train_fixed_point(
    model: nn.Module,
    *,
    length: int,
    temperature: float,
    delta_mu: float = 0.0,
    rounds: int = 10,
    batch_size: int = 64,
    gradient_steps: int = 10,
    learning_rate: float = 3e-3,
) -> list[float]:
    """Minimal generate-label-interpolate-regress fixed-point loop."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    for _ in range(rounds):
        terminals = sample_janus(model, batch_size, length, temperature, delta_mu)
        for _ in range(gradient_steps):
            losses.append(train_step(model, optimizer, terminals, temperature, delta_mu))
    return losses


__all__: Sequence[str] = (
    "JANUSIsing",
    "MaskedIsingConv",
    "energy",
    "exact_ising_observables",
    "ghost_wolff_samples",
    "heat_bath_labels",
    "heat_bath_prob",
    "ising_energy",
    "masked_soft_ce",
    "metrics",
    "observables",
    "reference_samples",
    "sample",
    "sample_janus",
    "train_fixed_point",
    "train_step",
    "wolff_step",
)
