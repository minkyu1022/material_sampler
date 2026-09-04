# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Stochastic differential equations and the bridge process used by BMS.

The reference process is a mean-free variance-exploding diffusion. Conditioning
it on its endpoints yields the Brownian bridge from which intermediate states are
sampled in closed form (see ``sample_posterior``), so full trajectories never need
to be stored.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from bms.utils.geometry import is_mean_free, subtract_mean


class BaseSDE(nn.Module, ABC):
    """Base class for SDEs ``dX_t = f(t, X_t) dt + g(t) dW_t``."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def drift(self, time: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        """Drift term of the SDE."""

    @abstractmethod
    def diffusion(self, time: torch.Tensor) -> torch.Tensor:
        """Diffusion term of the SDE."""

    def randn_like(self, data: torch.Tensor) -> torch.Tensor:
        """Generate random noise with the same shape as ``data``."""
        return torch.randn_like(data)


class MeanFreeZeroDriftSDE(BaseSDE, ABC):
    """Base class for mean-free SDEs with zero drift, ``dX_t = g(t) A dW_t``, where
    ``A`` projects onto the mean-free subspace."""

    def __init__(self):
        super().__init__()

    def drift(self, time: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(data)

    @abstractmethod
    def _diffsquare_integral(self, time: torch.Tensor | float) -> torch.DoubleTensor:
        """Integral of ``g(s)^2`` from ``0`` to ``time``."""

    @property
    def total_variance(self) -> torch.DoubleTensor:
        """Total variance accumulated over ``[0, 1]``."""
        return self._diffsquare_integral(1.0)

    def reparametrize_time(self, time: torch.Tensor) -> torch.DoubleTensor:
        """Reparametrize time according to the variance schedule."""
        return self._diffsquare_integral(time) / self.total_variance

    def randn_like(self, data: torch.Tensor) -> torch.Tensor:
        """Generate mean-free random noise with the same shape as ``data``."""
        return subtract_mean(super().randn_like(data))

    @torch.no_grad()
    def sample_posterior(
        self,
        time: torch.Tensor,  # [B]
        data_0: torch.Tensor,  # state at t=0, [B, N, 3]
        data_1: torch.Tensor,  # state at t=1, [B, N, 3]
        eps: float = 1e-10,
    ) -> torch.Tensor:
        """Sample ``X_t ~ p_{t|0,1}(X_t | X_0, X_1)`` from the Brownian bridge."""
        if data_0.shape != data_1.shape:
            raise ValueError("data_0 and data_1 must have the exact same shape.")
        if not is_mean_free(data_0) or not is_mean_free(data_1):
            raise ValueError("data_0 and data_1 must be mean-free (i.e. zero mean).")

        time_b = time.view(-1, 1, 1)  # broadcast against (B, N, 3)

        t_reparam = self.reparametrize_time(time_b)
        var = t_reparam * (1 - t_reparam) * self.total_variance
        std = torch.sqrt(var + eps).to(data_0)
        t_reparam = t_reparam.to(data_0)

        mean = (1 - t_reparam) * data_0 + t_reparam * data_1
        data_t = mean + std * self.randn_like(data_0)

        if not is_mean_free(data_t):
            raise ValueError("Sampled data_t is not mean-free")
        return data_t

    @torch.no_grad()
    def cond_score_t0(
        self,
        time: torch.Tensor,  # [B]
        data_0: torch.Tensor,  # state at t=0, [B, N, 3]
        data_t: torch.Tensor,  # state at t=t, [B, N, 3]
    ) -> torch.Tensor:
        """Compute the conditional score ``grad_{X_t} log p_{t|0}(X_t | X_0)``."""
        if data_0.shape != data_t.shape:
            raise ValueError("data_0 and data_t must have the exact same shape.")
        if not is_mean_free(data_0) or not is_mean_free(data_t):
            raise ValueError("data_0 and data_t must be mean-free (i.e. zero mean).")

        time_b = time.view(-1, 1, 1)
        var = self._diffsquare_integral(time_b)
        return (data_0 - data_t) / var.to(data_0)

    @torch.no_grad()
    def cond_var_t0(
        self,
        time: torch.Tensor,  # [B]
        data: torch.Tensor,  # state at t=0, [B, N, 3]
    ) -> torch.Tensor:
        """Compute the conditional variance ``kappa(time)``."""
        time_b = time.view(-1, 1, 1)
        var = self._diffsquare_integral(time_b)
        return var.to(data)

    @torch.no_grad()
    def cond_score_1t(
        self,
        time: torch.Tensor,  # [B]
        data_1: torch.Tensor,  # state at t=1, [B, N, 3]
        data_t: torch.Tensor,  # state at t=t, [B, N, 3]
    ) -> torch.Tensor:
        """Compute the conditional score ``grad_{X_t} log p_{1|t}(X_1 | X_t)``."""
        if data_1.shape != data_t.shape:
            raise ValueError("data_1 and data_t must have the exact same shape.")
        if not is_mean_free(data_1) or not is_mean_free(data_t):
            raise ValueError("data_1 and data_t must be mean-free (i.e. zero mean).")

        time_b = time.view(-1, 1, 1)
        var = self.total_variance - self._diffsquare_integral(time_b)
        return (data_1 - data_t) / var.to(data_1)


class VESDE(MeanFreeZeroDriftSDE):
    """Variance-exploding SDE."""

    def __init__(self, sigma_min: float = 0.001, sigma_max: float = 1.0):
        super().__init__()
        self.register_buffer("sigma_min", torch.as_tensor(sigma_min, dtype=torch.double))
        self.register_buffer("sigma_max", torch.as_tensor(sigma_max, dtype=torch.double))

    def diffusion(self, time: torch.Tensor) -> torch.Tensor:
        sigma_ratio = self.sigma_max / self.sigma_min
        diffusion = (
            self.sigma_min
            * (sigma_ratio ** (1 - time))
            * torch.sqrt(2 * torch.log(sigma_ratio))
        )
        return diffusion.to(time)

    def _diffsquare_integral(self, time: torch.Tensor | float) -> torch.DoubleTensor:
        time = torch.as_tensor(time, dtype=torch.double, device=self.sigma_max.device)
        sigma_ratio = self.sigma_max / self.sigma_min
        return (self.sigma_max ** 2) * (1 - sigma_ratio ** (-2 * time))


class EDMSDE(MeanFreeZeroDriftSDE):
    """EDM noise schedule from Karras et al. (2022)."""

    def __init__(
        self,
        sigma_min: float = 0.001,
        sigma_max: float = 1.0,
        rho: float = 7.0,
    ):
        super().__init__()
        self.register_buffer("sigma_min", torch.as_tensor(sigma_min, dtype=torch.double))
        self.register_buffer("sigma_max", torch.as_tensor(sigma_max, dtype=torch.double))
        self.register_buffer("rho", torch.as_tensor(rho, dtype=torch.double))

    def diffusion(self, time: torch.Tensor) -> torch.Tensor:
        return (
            (1 - time) * self.sigma_max ** (1 / self.rho)
            + time * self.sigma_min ** (1 / self.rho)
        ) ** self.rho

    def _diffsquare_integral(self, time: torch.Tensor | float) -> torch.DoubleTensor:
        time = torch.as_tensor(time, dtype=torch.double, device=self.sigma_max.device)
        return (
            self.sigma_max ** (2 + 1 / self.rho)
            - self.diffusion(time) ** (2 + 1 / self.rho)
        ) / (
            (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho))
            * (2 * self.rho + 1)
        )


class ControlledSDE(BaseSDE):
    """Controlled SDE with a learned control added to the drift,
    ``dX_t = (f(t, X_t) + g(t)^2 u(t, X_t)) dt + g(t) dW_t``."""

    def __init__(self, base_sde: BaseSDE, controller: nn.Module):
        super().__init__()
        self.base_sde = base_sde
        self.controller = controller

    def drift(
        self, time: torch.Tensor, data: torch.Tensor, return_control: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if time.dim() == 1:
            time = time.view(-1, 1, 1)

        diff = self.diffusion(time)
        control = diff * self.controller(time, data)
        drift = self.base_sde.drift(time, data) + diff * control
        if return_control:
            return drift, control
        return drift

    def diffusion(self, time: torch.Tensor) -> torch.Tensor:
        return self.base_sde.diffusion(time)
