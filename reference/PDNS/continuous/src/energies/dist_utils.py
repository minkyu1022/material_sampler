from __future__ import annotations

from typing import Dict
import logging
import math
from numbers import Number
import itertools

import torch
from torch import distributions
from torch.nn.init import trunc_normal_

import torchquad

# source: https://github.com/juliusberner/sde_sampler/tree/main/sde_sampler/distr

def sample_uniform(domain: torch.Tensor, batchsize: int = 1) -> torch.Tensor:
    dim = domain.shape[0]
    diam = domain[:, 1] - domain[:, 0]
    rand = torch.rand(batchsize, dim, device=domain.device)
    return domain[:, 0] + rand * diam


def rejection_sampling(
    shape: tuple, proposal: Distribution, target: Distribution, scaling: float
) -> torch.Tensor:
    n_samples = math.prod(shape)
    samples = proposal.sample((n_samples * math.ceil(scaling) * 10,))
    unif = torch.rand(samples.shape[0], 1, device=samples.device)
    unif *= scaling * proposal.pdf(samples)
    accept = unif < target.pdf(samples)
    samples = samples[accept]
    if samples.shape[0] >= n_samples:
        return samples[:n_samples].reshape(*shape, -1)
    else:
        new_shape = (n_samples - samples.shape[0],)
        new_samples = rejection_sampling(new_shape, proposal, target, scaling)
        return torch.concat([samples.reshape(*shape, -1), new_samples])


def gmm_params(name: str = "heart", dim: int = 2):
    if name == "heart":
        loc = 1.5 * torch.tensor(
            [
                [-0.5, -0.25],
                [0.0, -1],
                [0.5, -0.25],
                [-1.0, 0.5],
                [-0.5, 1.0],
                [0.0, 0.5],
                [0.5, 1.0],
                [1.0, 0.5],
            ]
        )
        factor = 1 / len(loc)

    elif name == "dist":
        loc = torch.tensor(
            [
                [0.0, 0.0],
                [2, 0.0],
                [0.0, 3.0],
                [-4, 0.0],
                [0.0, -5],
            ]
        )
        factor = math.sqrt(0.2)

    elif name in ["fab", "multi"]:
        n_mixes, loc_scaling = (40, 40) if name == "fab" else (80, 80)
        generator = torch.Generator()
        generator.manual_seed(42)
        # loc = (torch.rand((n_mixes, 2), generator=generator) - 0.5) * 2 * loc_scaling
        loc =  [[-0.2995, 21.4577], [-32.9218, -29.4376],
                [-15.4062, 10.7263], [-0.7925, 31.7156],
                [-3.5498, 10.5845], [-12.0885, -7.8626],
                [-38.2139, -26.4913], [-16.4889, 1.4817],
                [15.8134, 24.0009], [-27.1176, -17.4185],
                [14.5287, 33.2155], [-8.2320, 29.9325],
                [-6.4473, 4.2326], [36.2190, -37.1068],
                [-25.1815, -10.1266], [-15.5920, 34.5600],
                [-25.9272, -18.4133], [-27.9456, -37.4624],
                [-23.3496, 34.3839], [17.8487, 19.3869],
                [2.1037, -20.5073], [6.7674, -37.3478],
                [-28.9026, -20.6212], [25.2375, 23.4529],
                [-17.7398, -1.4433], [25.5824, 39.7653],
                [15.8753, 5.4037], [26.8195, -23.5521],
                [7.4538, -31.0122], [-27.7234, -20.6633],
                [18.0989, 16.0864], [-23.6941, 12.0843],
                [21.9589, -5.0487], [1.5273, 9.2682],
                [24.8151, 38.4078], [-30.8249, -14.6588],
                [15.7204, 33.1420], [34.8083, 35.2943],
                [7.9606, -34.7833], [3.6797, -25.0242]]
        loc = torch.tensor(loc).float()
        factor = torch.nn.functional.softplus(torch.tensor(1.0, device=loc.device))

    elif name == "grid":
        x_coords = torch.linspace(-5, 5, 3)
        loc = torch.cartesian_prod(x_coords, x_coords)
        factor = math.sqrt(0.3)

    elif name == "circle":
        freq = 2 * torch.pi * torch.arange(1, 9) / 8
        loc = torch.stack([4.0 * freq.cos(), 4.0 * freq.sin()], dim=1)
        factor = math.sqrt(0.3)

    else:
        raise ValueError("Unknown mode for the Gaussian mixture.")


    if dim > 2:
        loc = torch.cat([loc, torch.zeros(loc.shape[0], dim - 2)], dim=1)
    scale = factor * torch.ones_like(loc)
    mixture_weights = torch.ones(loc.shape[0], device=loc.device)
    return loc, scale, mixture_weights


class Distribution(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        log_norm_const: float = None,
        domain: float | torch.Tensor | None = None,
        n_reference_samples: int | None = None,
        grid_points: int | None = None,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.n_reference_samples = n_reference_samples
        self.grid_points = grid_points
        self.set_domain(domain)

        # Initialize
        self.log_norm_const = log_norm_const
        self.register_buffer("stddevs", None, persistent=False)
        self.expectations = {}

        self.name = self.__class__.__name__

    def set_domain(self, d: torch.Tensor | float | None = None):
        if d is not None:
            if not isinstance(d, torch.Tensor):
                d = torch.tensor(d, dtype=torch.float)
            if d.ndim == 0:
                d = torch.stack([-d, d], dim=-1)
            if d.ndim == 1:
                d = d.unsqueeze(0)
            if d.shape == (1, 2):
                d = d.repeat(self.dim, 1)
            assert d.shape == (self.dim, 2)
        self.register_buffer("domain", d, persistent=False)

    def _initialize_distr(self):
        # This can be used to reinitialize distributions, e.g,  when transfering between devices
        pass

    def _apply(self, fn):
        torch.nn.Module._apply(self, fn)
        self._initialize_distr()

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        if self.log_norm_const is None:
            raise NotImplementedError
        return self.unnorm_log_prob(x) - self.log_norm_const

    def pdf(self, x: torch.Tensor) -> torch.Tensor:
        return self.log_prob(x).exp()

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def unnorm_pdf(self, x: torch.Tensor) -> torch.Tensor:
        return self.unnorm_log_prob(x).exp()

    def score(self, x: torch.Tensor, create_graph=False) -> torch.Tensor:
        grad = x.requires_grad
        x.requires_grad_(True)
        with torch.set_grad_enabled(True):
            log_rho = self.unnorm_log_prob(x).sum()
            score = torch.autograd.grad(log_rho, x, create_graph=create_graph)[0]
        x.requires_grad_(grad)
        return score

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unnorm_log_prob(x)


class GMM(Distribution):
    def __init__(
        self,
        dim: int = 2,
        loc: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        mixture_weights: torch.Tensor | None = None,
        n_reference_samples: int = int(1e7),
        example: str | None = None,
        log_norm_const: float = 0.0,
        domain_scale: float = 5,
        domain_tol: float | None = 1e-5,
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            log_norm_const=log_norm_const,
            n_reference_samples=n_reference_samples,
            **kwargs,
        )
        if example is not None:
            if any(t is not None for t in [loc, scale, mixture_weights]):
                logging.warning(
                    "Ignoring loc, scale, and mixture weights since name is specified."
                )
            loc, scale, mixture_weights = gmm_params(example, dim=dim)

        # Check shapes
        n_mixtures = loc.shape[0]
        if not loc.shape == scale.shape == (n_mixtures, self.dim):
            raise ValueError("Shape missmatch between loc and scale.")
        if mixture_weights is None and n_mixtures > 1:
            raise ValueError("Require mixture weights.")
        if not (mixture_weights is None or mixture_weights.shape == (n_mixtures,)):
            raise ValueError("Shape missmatch for the mixture weights.")

        # Initialize
        try: device = kwargs['device']
        except: device = 'cpu'
        self.register_buffer("loc", loc.to(device), persistent=False)
        self.register_buffer("scale", scale.to(device), persistent=False)
        self.register_buffer("mixture_weights", mixture_weights, persistent=False)
        self._initialize_distr()

        # Check domain
        if self.domain is None:
            deviation = domain_scale * self.scale.max(dim=0).values
            deviation = torch.stack([-deviation, deviation], dim=-1)
            pos = torch.stack(
                [self.loc.min(dim=0).values, self.loc.max(dim=0).values], dim=-1
            )
            self.set_domain(pos + deviation)
        if domain_tol is not None and (self.pdf(self.domain.T) > domain_tol).any():
            raise ValueError("domain does not satisfy tolerance at the boundary.")

        # set name for logging
        self.name = "gmm"

    @property
    def stddevs(self) -> torch.Tensor:
        return self.distr.variance.sqrt()

    def _initialize_distr(
        self,
    ) -> distributions.MixtureSameFamily | distributions.Independent:
        if self.mixture_weights is None:
            self.distr = distributions.Independent(
                distributions.Normal(self.loc.squeeze(0), self.scale.squeeze(0)), 1
            )
        else:
            modes = distributions.Independent(
                distributions.Normal(self.loc, self.scale), 1
            )
            mix = distributions.Categorical(self.mixture_weights)
            self.distr = distributions.MixtureSameFamily(mix, modes)

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        log_prob = self.distr.log_prob(x).unsqueeze(-1) + self.log_norm_const
        assert log_prob.shape == (*x.shape[:-1], 1)
        return log_prob

    def marginal_distr(self, dim=0) -> torch.distributions.Distribution:
        if self.mixture_weights is None:
            return distributions.Normal(self.loc[0, dim], self.scale[0, dim])
        modes = distributions.Normal(self.loc[:, dim], self.scale[:, dim])
        mix = distributions.Categorical(self.mixture_weights)
        return distributions.MixtureSameFamily(mix, modes)

    def marginal(self, x: torch.Tensor, dim=0) -> torch.Tensor:
        return self.marginal_distr(dim=dim).log_prob(x).exp()

    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        if shape is None:
            shape = tuple()
        return self.distr.sample(torch.Size(shape))


class GMM40(GMM):
    def __init__(
        self,
        dim: int = 2,
        num_components: int = 40,
        loc_scaling: float = 40,
        grid_parameter: int = 0,
        scale_scaling: float = 1.0,
        seed: int = 0,
        sample_bounds=None,
        can_sample=True,
        log_Z: float = 0,
    ):
        if grid_parameter != 0:
            num_components = 3**dim

        self.seed = seed
        self.n_mixes = num_components

        generator = torch.Generator().manual_seed(seed)

        if grid_parameter == 0:
            mean = (
                torch.rand((num_components, dim), generator=generator) * 2 - 1
            ) * loc_scaling
            scale = torch.ones(num_components, dim) * scale_scaling
        else:
            coords = [-grid_parameter, 0, grid_parameter]
            mean = torch.tensor(list(itertools.product(coords, repeat=dim)), dtype=torch.float)
            scale = torch.ones(mean.shape) * scale_scaling
            
        super().__init__(dim=dim, log_norm_const=log_Z, loc=mean, scale=scale, mixture_weights=torch.ones(num_components))

    def entropy(self, samples: torch.Tensor) -> torch.Tensor:
        expanded = samples.unsqueeze(-2)
        log_probs = self.distr.component_dist.log_prob(expanded)
        idx = torch.argmax(log_probs, dim=1)
        unique_elements, counts = torch.unique(idx, return_counts=True)
        mode_dist = counts.float() / samples.shape[0]
        entropy = -torch.sum(mode_dist * (torch.log(mode_dist) / torch.log(torch.tensor(self.n_mixes, dtype=torch.float))))
        return entropy


class Gauss(GMM):
    def __init__(
        self,
        dim: int = 1,
        loc: torch.Tensor | Number = 0.0,
        scale: torch.Tensor | Number = 1.0,
        **kwargs,
    ):

        # Setup parameters
        params = {"loc": loc, "scale": scale}
        params = {k: Gauss._prepare_input(p, dim) for k, p in params.items()}
        super().__init__(dim=dim, **params, **kwargs)
        self.stddevs = self.scale.squeeze(0)

        # set name for logging
        self.name = "non-iso-gauss"

    @staticmethod
    def _prepare_input(param: torch.Tensor | Number, dim: int = 1):
        if not isinstance(param, torch.Tensor):
            param = torch.tensor(param, dtype=torch.float)
        param = torch.atleast_2d(param)
        if param.numel() == 1:
            param = param.repeat(1, dim)
        return param

    def score(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return (self.loc - x) / self.scale**2


class IsotropicGauss(Gauss):
    # Typially used as prior (supports truncation and faster methods)
    def __init__(
        self,
        dim: int = 1,
        loc: float = 0.0,
        scale: float = 1.0,
        truncate_quartile: float | None = None,
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            loc=loc,
            scale=scale,
            **kwargs,
        )

        assert torch.allclose(self.loc, self.loc[0, 0])
        assert torch.allclose(self.scale, self.scale[0, 0])

        # Calculate truncation values
        if truncate_quartile is not None:
            quartiles = torch.tensor(
                [truncate_quartile / 2, 1 - truncate_quartile / 2],
                device=self.domain.device,
            )
            truncate_quartile = self.marginal_distr().icdf(quartiles).tolist()
        self.truncate_quartile = truncate_quartile

        # set name for logging
        self.name = "gauss"

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        var = self.scale[0, 0] ** 2
        norm_const = -0.5 * self.dim * (2.0 * math.pi * var).log()
        norm_const += self.log_norm_const
        sq_sum = torch.sum((x - self.loc[0, 0]) ** 2, dim=-1, keepdim=True)
        return norm_const - 0.5 * sq_sum / var

    def score(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return (self.loc[0, 0] - x) / self.scale[0, 0] ** 2

    def marginal(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.marginal_distr().log_prob(x).exp()

    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        if shape is None:
            shape = tuple()
        if self.truncate_quartile is None:
            return self.loc[0, 0] + self.scale[0, 0] * torch.randn(
                *shape, self.dim, device=self.domain.device
            )
        tensor = torch.empty(*shape, self.dim, device=self.domain.device)
        return trunc_normal_(
            tensor,
            mean=self.loc[0, 0],
            std=self.scale[0, 0],
            a=self.truncate_quartile[0],
            b=self.truncate_quartile[1],
        )


class Delta(Gauss):
    def __init__(
        self,
        dim: int = 1,
        loc: torch.Tensor | float = 0.0,
        approx_scale: float = 1e-3,
        domain_scale: float = 10,
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            loc=loc,
            scale=approx_scale,
            domain_scale=domain_scale,
            **kwargs,
        )

        # set name for logging
        self.name = "delta"

    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        if shape is None:
            shape = tuple()
        return self.loc.repeat(*shape, 1)


class Funnel(Distribution):
    def __init__(
        self,
        dim: int = 10,
        variance: float | None = None,
        n_reference_samples: int = int(1e7),
        log_norm_const: float = 0.0,
        domain_first_scale: float = 5.0,
        domain_other_scale: float = 5.0,
        domain_tol: float | None = 1e-5,
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            log_norm_const=log_norm_const,
            n_reference_samples=n_reference_samples,
            **kwargs,
        )
        self.variance = variance
        if self.variance is None:
            self.variance = self.dim - 1
        self.distr_first = IsotropicGauss(
            dim=1,
            scale=math.sqrt(self.variance),
            domain_scale=domain_first_scale,
            domain_tol=domain_tol,
        )
        self._initialize_distr()

        # Check domains
        if self.domain is None:
            domain_other = (
                self.distr_first.domain.sgn()
                * (self.distr_first.domain.abs() / domain_other_scale).exp()
            )
            self.set_domain(
                torch.cat(
                    [self.distr_first.domain, domain_other.repeat(self.dim - 1, 1)]
                )
            )
        if domain_tol is not None and (self.pdf(self.domain.T) > domain_tol).any():
            raise ValueError("Domain does not satisfy tolerance at the boundary.")

        # set name for logging
        self.name = "funnel"

    @staticmethod
    def log_prob_other(x_other, x_first):
        norm_const = -x_other.shape[-1] * (x_first + math.log(2.0 * math.pi)) / 2.0
        x_sq_sum = (x_other**2).sum(dim=-1, keepdim=True)
        return norm_const - 0.5 * x_sq_sum * (-x_first).exp()

    def _initialize_distr(self):
        self.distr_first._initialize_distr()

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        x_first = x[:, 0].unsqueeze(-1)
        log_prob_first = self.distr_first.unnorm_log_prob(x_first)
        log_prob_other = Funnel.log_prob_other(x[:, 1:], x_first)
        assert log_prob_other.shape == log_prob_first.shape == (x.shape[0], 1)
        log_prob = log_prob_first + log_prob_other
        return log_prob + self.log_norm_const

    def score(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        assert x.ndim == 2 and x.shape[-1] == self.dim
        x_first = x[:, 0].unsqueeze(-1)
        x_other = x[:, 1:]
        inv_var_other = (-x_first).exp()
        score_first = self.distr_first.score(x_first) - 0.5 * x_other.shape[-1]
        score_first += 0.5 * (x_other**2).sum(dim=-1, keepdim=True) * inv_var_other
        assert score_first.shape == (x.shape[0], 1)
        score_other = -x_other * inv_var_other
        return torch.cat([score_first, score_other], dim=-1)

    def marginal(self, x: torch.Tensor, dim=0) -> torch.Tensor:
        assert x.ndim == 2 and x.shape[-1] == 1
        if dim == 0:
            return self.distr_first.marginal(x)
        samples_first = self.distr_first.sample((self.n_reference_samples, 1))
        log_prob = self.log_prob_other(x, samples_first)
        return log_prob.exp().mean(axis=0)

    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        if shape is None:
            shape = tuple()
        samples_first = self.distr_first.sample(shape)
        stdd_other = (0.5 * samples_first).exp()
        samples_other = torch.randn(*shape, self.dim - 1, device=samples_first.device)
        return torch.cat((samples_first, samples_other * stdd_other), dim=-1)


class DoubleWell(Distribution):
    def __init__(
        self,
        dim: int = 1,
        separation: float = 2.0,
        shift: float = 0.0,
        grid_points: int = 2001,
        rejection_sampling_scaling: float = 3.0,
        domain_delta: float = 2.5,
        **kwargs,
    ):
        if not dim == 1:
            raise ValueError("`dim` needs to be `1`. Consider using `MultiWell`.")
        super().__init__(dim=1, grid_points=grid_points, **kwargs)
        self.rejection_sampling_scaling = rejection_sampling_scaling
        self.register_buffer("separation", torch.tensor(separation), persistent=False)
        self.register_buffer("shift", torch.tensor(shift), persistent=False)

        # Set domain
        if self.domain is None:
            domain = self.shift + (
                self.separation.sqrt() + domain_delta
            ) * torch.tensor([[-1.0, 1.0]])
            self.set_domain(domain)

        # set name for logging
        self.name = "double-well"

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.shift
        return -((x**2 - self.separation) ** 2)

    def score(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = x - self.shift
        return -4.0 * (x**2 - self.separation) * x

    def marginal(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.pdf(x)

    def get_proposal_distr(self):
        device = self.domain.device
        loc = self.shift + self.separation.sqrt() * torch.tensor(
            [[-1.0], [1.0]], device=device
        )
        scale = 1 / self.separation.sqrt() * torch.ones(2, 1, device=device)
        proposal = GMM(
            dim=1,
            loc=loc,
            scale=scale,
            mixture_weights=torch.ones(2, device=device),
            domain_tol=None,
        )
        proposal.to(device)
        return proposal

    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        if shape is None:
            shape = tuple()
        proposal = self.get_proposal_distr()
        return rejection_sampling(
            shape=shape,
            target=self,
            proposal=proposal,
            scaling=self.rejection_sampling_scaling,
        )


class MultiWell(Distribution):
    def __init__(
        self,
        dim: int = 2,
        n_double_wells: int = 1,
        separation: float = 2.0,
        shift: float = 0.0,
        domain_dw_delta: float = 2.5,
        domain_gauss_scale: float = 5.0,
        **kwargs,
    ):
        super().__init__(dim=dim, **kwargs)
        # Define parameters
        self.separation = separation
        if n_double_wells > dim or n_double_wells == 0:
            raise ValueError(f"Please specify between 1 and {dim} double wells.")
        self.n_double_wells = n_double_wells
        self.n_gauss = self.dim - self.n_double_wells

        # Initialize distributions
        self.double_well = DoubleWell(
            separation=separation, shift=shift, domain_delta=domain_dw_delta
        )
        domain = self.double_well.domain.repeat(self.n_double_wells, 1)
        self.gauss = None
        if self.n_gauss > 0:
            self.gauss = IsotropicGauss(
                dim=self.n_gauss,
                loc=shift,
                log_norm_const=0.5 * math.log(2.0 * math.pi) * self.n_gauss,
                domain_scale=domain_gauss_scale,
            )
            domain = torch.cat([domain, self.gauss.domain])

        # Set domain
        self.set_domain(domain)

        # set name for logging
        self.name = "multi-well"

    def _initialize_distr(self):
        if self.gauss is not None:
            return self.gauss._initialize_distr()

    def compute_stats(self):
        # Double well
        self.double_well.compute_stats()
        self.log_norm_const = self.double_well.log_norm_const * self.n_double_wells
        self.expectations = {
            name: exp * self.n_double_wells
            for name, exp in self.double_well.expectations.items()
        }
        self.stddevs = torch.cat([self.double_well.stddevs] * self.n_double_wells)

        # Gauss
        if self.gauss is not None:
            self.gauss.compute_stats()
            self.log_norm_const += self.gauss.log_norm_const
            for name in self.expectations:
                # This assumes that the expectations are reducing the dim via a sum
                self.expectations[name] += self.gauss.expectations[name]
            self.stddevs = torch.cat([self.stddevs, self.gauss.stddevs])

        assert (self.pdf(self.domain.T) < 1e-5).all()

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        log_prob = self.double_well.unnorm_log_prob(x[:, : self.n_double_wells]).sum(
            dim=-1, keepdim=True
        )
        if self.gauss is not None:
            log_prob += self.gauss.unnorm_log_prob(x[:, self.n_double_wells :])
        assert log_prob.shape == (*x.shape[:-1], 1)
        return log_prob

    def score(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        score = self.double_well.score(x[:, : self.n_double_wells])
        if self.gauss is not None:
            score_gauss = self.gauss.score(x[:, self.n_double_wells :])
            score = torch.cat([score, score_gauss], dim=-1)
        return score

    def marginal(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        if dim < self.n_double_wells:
            return self.double_well.marginal(x)
        return self.gauss.marginal(x)

    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        if shape is None:
            shape = tuple()
        samples = self.double_well.sample(shape + (self.n_double_wells,)).squeeze(-1)
        if self.gauss is not None:
            samples_gauss = self.gauss.sample(shape)
            samples = torch.cat([samples, samples_gauss], dim=-1)
        return samples


class Rings(Distribution):
    def __init__(
        self,
        dim: int = 2,
        lower_rad: float = 1.0,
        upper_rad: float = 5.0,
        num_rad: int = 3,
        scale: float = 100.0,
        grid_points: int = 2001**2,
        scale_domain: float = 10.0,
        domain_tol: float | None = 1e-5,
        eps: float = 1e-8,
        **kwargs,
    ):
        if dim != 2:
            raise ValueError("The rings should be two-dimensional.")
        super().__init__(dim=dim, grid_points=grid_points, **kwargs)
        self.register_buffer(
            "r_centers", torch.linspace(lower_rad, upper_rad, num_rad), persistent=False
        )
        self.scale = scale
        self.eps = eps

        # Set domain
        self.domain_tol = domain_tol
        if self.domain is None:
            self.set_domain(
                self.r_centers.max() + scale_domain / math.sqrt(self.scale / 2)
            )

        # set name for logging
        self.name = "rings"

    def compute_stats(self):
        super().compute_stats()
        if (
            self.domain_tol is not None
            and (self.pdf(self.domain.T) > self.domain_tol).any()
        ):
            raise ValueError("Domain does not satisfy tolerance at the boundary.")

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        radius = torch.norm(x, dim=-1, keepdim=True)
        return (
            -self.scale
            * (radius - self.r_centers).square().min(dim=-1, keepdim=True).values
        )

    def score(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        radius = torch.norm(x, dim=-1, keepdim=True)
        indices = (radius - self.r_centers).square().min(dim=-1).indices
        centers = self.r_centers[indices].unsqueeze(-1)
        assert centers.shape == (x.shape[0], 1)
        return -2.0 * self.scale * (1 - centers / (radius + self.eps)) * x

    def _integrand(
        self, y: torch.Tensor, x: torch.Tensor, dim: int = 0
    ) -> torch.Tensor:
        x = x.tile(y.shape[0], 1, 1)
        y = y.unsqueeze(1).tile(1, x.shape[1], 1)
        assert y.shape == x.shape
        if dim == 0:
            tensor = torch.cat([x, y], dim=-1)
        else:
            tensor = torch.cat([y, x], dim=-1)
        return self.pdf(tensor).squeeze(-1)

    def marginal(self, x: torch.Tensor, dim: int = 0) -> torch.Tenor:
        integrator = torchquad.Boole()
        with torch.device(self.domain.device):
            domain = self.domain[dim].unsqueeze(0)
            integral = integrator.integrate(
                lambda y: self._integrand(y, x, dim=dim),
                dim=1,
                N=2001,
                integration_domain=domain,
            )
        return integral.unsqueeze(-1)


# https://github.com/jarridrb/DEM/blob/main/dem/energies/base_prior.py#L24
class CenteredParticlesGauss(distributions.Distribution):
    """ sample particles with zero center of mass
    """
    arg_constraints: Dict[str, distributions.constraints.Constraint] = {}

    def __init__(self, n_particles, spatial_dim, scale, device="cpu", **kwargs):
        super().__init__()
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim
        self.dim = n_particles * spatial_dim
        self.scale = scale
        self.device = device
        self.name = "meanfree"

    def log_prob(self, x):
        x = x.reshape(-1, self.n_particles, self.spatial_dim)
        N, D = x.shape[-2:]

        # r is invariant to a basis change in the relevant hyperplane.
        r2 = torch.sum(x**2, dim=(-1, -2)) / self.scale**2

        # The relevant hyperplane is (N-1) * D dimensional.
        degrees_of_freedom = (N - 1) * D

        # Normalizing constant and logpx are computed:
        log_normalizing_constant = (
            -0.5 * degrees_of_freedom * math.log(2 * torch.pi * self.scale**2)
        )
        log_px = -0.5 * r2 + log_normalizing_constant
        return log_px

    def unnorm_log_prob(self, x):
        return self.log_prob(x)

    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        if shape is None:
            shape = tuple()
        samples = torch.randn(*shape, self.dim, device=self.device) * self.scale
        samples = samples.reshape(-1, self.n_particles, self.spatial_dim)
        samples = samples - samples.mean(-2, keepdims=True)
        return samples.reshape(*shape, self.n_particles * self.spatial_dim)
    
    def score(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return - x / self.scale**2


class HarmonicCenteredGauss(distributions.Distribution):
    """ sample particles with zero COM from non-isotropic Gaussian based on
        a harmonic prior

        https://arxiv.org/pdf/2304.02198
    """
    arg_constraints: Dict[str, distributions.constraints.Constraint] = {}

    def __init__(self, n_particles, spatial_dim, scale, device="cpu", **kwargs):
        super().__init__()
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim
        self.dim = n_particles * spatial_dim
        self.scale = scale
        self.device = device
        self.name = "harmonic"

        cov = self._compute_cov(n_particles, spatial_dim)
        # print(cov)
        self.rank, self.A = self._decompose_svd(cov)

    def _compute_cov(self, n_particles, spatial_dim):
        """ e.g., n_particles = 2, spatial_dim = 3 would generate

            R = tensor([[  1,   0,   0, -0.5,   0,   0 ],
                        [  0,   1,   0,   0, -0.5,   0 ],
                        [  0,   0,   1,   0,    0, -0.5],
                        [-0.5,  0,   0,   1,    0,   0 ],
                        [  0, -0.5,   0,  0,    1,   0 ],
                        [  0,   0, -0.5,  0,    0,   1 ]])

            Denote x = [a1, a2, a3, b1, b2, b3]. This yields

            0.5 * x^T R x = a**2 + b**2 - ab = (a - b)**2
        """
        # TODO(ghliu) assume all particles are connected; otherwise changes A
        A = - 0.5 * torch.ones(n_particles, n_particles)
        A[torch.arange(n_particles), torch.arange(n_particles)] = 1.
        B = torch.eye(spatial_dim)
        return torch.kron(A, B)

    def _decompose_svd(self, cov):
        """ return the rank of cov and A where `cov = UΣV = AA^T`
        """
        U, S, Vt = torch.svd(cov)

        rank = (S > 1e-8).sum()
        A = U[:, :rank] @ torch.diag(S[:rank]).sqrt()
        return rank, A

    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        """ generate samples by
            1. z ~ N(0,I) in the subspace with dim=rank
            2. x = Az, hence x ~ N(0, AA^T)
            3. make x zero COM
        """
        if shape is None:
            shape = tuple()

        B = math.prod(shape) # batch
        z = torch.randn(B, self.rank, device=self.device)
        samples = z @ self.A.to(z).T # (B, R) x (R, D) = (B, D)
        assert samples.shape == (B, self.dim)

        samples = samples * self.scale # note: scale = sqrt(alpha)

        samples = samples.reshape(*shape, self.n_particles, self.spatial_dim)
        samples = samples - samples.mean(-2, keepdims=True)
        return samples.reshape(*shape, self.dim)