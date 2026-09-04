from __future__ import annotations
import numpy as np
from typing import Callable, List
from tqdm import tqdm
import torch

from mace.tools.torch_geometric import Batch

from src.energies.dist_utils import CenteredParticlesGauss
import src.utils.graph_utils as graph_utils

from ipdb import set_trace as debug

class BaseSDE(torch.nn.Module):
    """ dX_t = f(t, X_t) dt + g(t) dW_t
    """
    def __init__(self):
        super().__init__()
        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def register(self, name: str, val: float):
        self.register_buffer(
            name,
            torch.tensor(val, dtype=torch.float),
            persistent=False,
        )

    @property
    def has_drift(self) -> bool:
        return True

    def randn_like(self, x: torch.Tensor):
        return torch.randn_like(x)

    def propagate(self, x, dx):
        return x + dx

    def _pt_gauss_param(
        self,
        t: torch.Tensor,
        mu0: torch.Tensor,
        var0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ time-marginal p_t(x) as a Gaussian given p0 = N(mu0, var0)
        """
        raise NotImplementedError

    def pt_gauss_param(self, t, mu0, var0):
        # dump func for graph assertion
        return self._pt_gauss_param(t, mu0, var0)

    def cond_score(self, x0: torch.Tensor, t: torch.Tensor, xt: torch.Tensor):
        """ p_{t|0}(x|x0) = N(x; μ, Σ) as a Gaussian
            ∇log p = (μ - x) / Σ
        """
        loc, var = self._pt_gauss_param(t, x0)
        return (loc - xt) / var

    # f
    def drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # g
    def diff(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.drift(t, x)
    
    def g(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.diff(t) * torch.ones_like(x)


class ConstOrnsteinUhlenbeckSDE(BaseSDE):
    """ dX_t = a X_t dt + σ dW_t
        dμ_t = a μ_t dt         , μ(0) = μ_0 ---> μ(t) = μ_0 exp(at)
        dΣ_t = [2a Σ_t + σ^2] dt, Σ(0) = Σ_0 ---> Σ(t) = Σ_0 exp(2at) - σ^2 (1 - exp(2at)) / (2a)
    """
    def __init__(self, a: float = 2.0, sigma: float = 1.0):
        super().__init__()
        # TODO(ghliu) adopt from sde_sampler, impl should work for all a != 0
        assert a >= 0
        self.register("a", a)
        self.a: torch.Tensor

        assert sigma > 0
        self.register("sigma", sigma)
        self.sigma: torch.Tensor

    def drift(self, t, x):
        return self.a * x

    def diff(self, t):
        return torch.full_like(t, self.sigma)

    def _pt_gauss_param(
        self,
        t: torch.Tensor,
        mu0: torch.Tensor,
        var0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO(ghliu) check dim

        a, sigma = self.a, self.sigma
        coeff = torch.exp(a * t)
        coeff2 = torch.exp(a * t * 2)

        mu_t = coeff * mu0
        var_t = (
            -sigma**2
            / (2 * a)
            * (1 - coeff2)
        )
        if var0 is not None:
            var_t = var_t + coeff2 * var0
        return mu_t, var_t


## Basic BM, VE, VP ##
class BrownianMotionSDE(ConstOrnsteinUhlenbeckSDE):
    """ dX_t = σ dW_t
        dμ_t = 0 dt  , μ(0) = μ_0 ---> μ(t) = μ_0
        dΣ_t = σ^2 dt, Σ(0) = Σ_0 ---> Σ(t) = Σ_0 + σ^2 t
    """
    def __init__(self, sigma: float = 2.0):
        super().__init__(a=0.0, sigma=sigma)

    @property
    def has_drift(self) -> bool:
        return False


    def sample_posterior(self, t, x0, x1, expand_dim=False):
        """ expand_dim = True
            t: (T, 1)  x0: (B, D)  x1: (B, D)
            return: xt: (B, T, D)

            expand_dim = False
            t: (B, 1)  x0: (B, D)  x1: (B, D)
            return: xt: (B, D)
        """
        (B, D), T = x0.shape, t.shape[0]
        assert x1.shape == (B, D) and t.shape == (T, 1)
        if not expand_dim: assert B == T

        if expand_dim:
            tt = t[None, :] # (1, T, 1)
            mean = (1 - tt) * x0.unsqueeze(1) + tt * x1.unsqueeze(1)
            var = (1 - tt) * tt * self.sigma**2
            var[var < 0] = 0
            noise = var.sqrt() * self.randn_like(mean)
            assert mean.shape == noise.shape == (B, T, D)

        else:
            mean = (1 - t) * x0 + t * x1
            var = (1 - t) * t * self.sigma**2
            var[var < 0] = 0
            noise = var.sqrt() * self.randn_like(mean)
            assert mean.shape == noise.shape == (B, D)

        xt = mean + noise

        return xt, - (xt - x1) / ((1 - t + 1e-6) * self.sigma**2), (1 - t) * self.sigma**2


class VESDE(BaseSDE):
    def __init__(self, sigma_min, sigma_max):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_diff = sigma_max / sigma_min
        self.total_var = sigma_max**2 - sigma_min**2

    @property
    def has_drift(self) -> bool:
        return False

    def _diffsquare_integral(self, t):
        '''
        integral g^2(t) from 0 to t
        Note that integral g^2(t) from 0 to 1 is sigma_max^2 - sigma_min^2
        '''
        return (self.sigma_max**2) * (1 - (self.sigma_diff) ** (-2 * t))

    def drift(self, t, x):
        return torch.zeros_like(x)

    def diff(self, t):
        return self.sigma_min * (self.sigma_diff ** (1-t)) * ((2 * np.log(self.sigma_diff)) ** 0.5)

    def _pt_gauss_param(self, t, mu0, var0 = None):
        var = self._diffsquare_integral(t)
        if var0 is not None:
            var = var + var0
        return mu0, var

    def sample_posterior(self, t, x0, x1, z=None):
        """ t: (B, 1)  x0: (B, D)  x1: (B, D)
            return: xt: (B, D)
        """
        (B, D) = x0.shape
        assert x1.shape == (B, D) and t.shape == (B, 1)
        # print(self._diffsquare_integral(t))
        t_reparam = self._diffsquare_integral(t) / self.total_var
        if z is None:
            z = self.randn_like(x0)
        assert z.shape == (B, D)

        mean = (1 - t_reparam) * x0 + t_reparam * x1
        var = self.total_var * t_reparam * (1 - t_reparam)
        var[var < 0] = 0 # NOTE: avoid numerical error close to boundary
        noise = torch.sqrt(var) * z
        assert mean.shape == noise.shape == (B, D)

        xt = mean + noise
        return xt, - (xt - x1) / (self.total_var * (1-t_reparam + 1e-6)), self.total_var * (1-t_reparam)


# VPSDE
class VPSDE(BaseSDE):
    """ Special case of LangevinSDE when
                 p(t, x) = N(0, σ^2)
            ∇log p(t, x) = -x / σ^2
        which yields...
            dX_t = - β_t / 2 * X_t dt + σ^2 sqrt(β_t) dW_t

        Note: if X_0 ~ N(0, σ^2), then X_t ~ N(0, σ^2) for all t ∈ [0,1]

    """
    def __init__(
        self,
        beta0: float = 20.0,
        beta1: float = 0.1,
        sigma: float = 1.0,
    ):
        super().__init__()

        self.register_buffer(
            "beta1",
            torch.tensor(beta1, dtype=torch.float),
            persistent=False,
        )
        self.beta1: torch.Tensor
        self.register_buffer(
            "beta0",
            torch.tensor(beta0, dtype=torch.float),
            persistent=False,
        )
        self.beta0: torch.Tensor
        self.register_buffer(
            "sigma",
            torch.tensor(sigma, dtype=torch.float),
            persistent=False,
        )

    def _beta(self, t: torch.Tensor) -> torch.Tensor:
        return torch.lerp(self.beta0, self.beta1, t)

    def drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return - 0.5 * self._beta(t) * x

    def diff(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma * torch.sqrt(self._beta(t))

    def score(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return - 1. / (self.sigma ** 2) * x

    def coeff2(self, t):
        bt, b1 = self._beta(t), self.beta1
        return -0.25 * (1-t) * (bt + b1) # = $ -\\frac{1}{2} \int^1_t \\beta(s) ds$

    def sample_posterior(self, t, x0, x1, expand_dim=False):
        bt, b0, b1 = self._beta(t), self.beta0, self.beta1
        coeff1 = -0.25 * t * (bt + b0) # = $ -\\frac{1}{2} \int^t_0 \\beta(s) ds$
        coeff2 = -0.25 * (1-t) * (bt + b1) # = $ -\\frac{1}{2} \int^1_t \\beta(s) ds$
        coeff3 = -0.25 * (b1 + b0)

        mu = torch.exp(coeff1) * (1 - torch.exp(2*coeff2)) / (1 - torch.exp(2*coeff3)) * x0 \
           + torch.exp(coeff2) * (1 - torch.exp(2*coeff1)) / (1 - torch.exp(2*coeff3)) * x1

        var = (1 - torch.exp(2*coeff1)) * (1 - torch.exp(2*coeff2)) / (1 - torch.exp(2*coeff3) + 1e-8)
        var[var < 0] = 0
        std = self.sigma * torch.sqrt(var)

        z = torch.randn_like(mu)
        xt = mu + std * z

        
        score = - torch.exp(coeff2) * (torch.exp(coeff2) * xt - x1) / (self.sigma**2 * (1 - torch.exp(2*coeff2)) + 1e-6)
        # score = - (torch.exp(coeff2) * xt - x1) / (self.sigma**2 * (1 - torch.exp(2*coeff2)) + 1e-6)
        return xt, score, self.sigma**2 * (1 - torch.exp(2*coeff2))
    

# AnnealedSDE (when we use VPSDE)
class AnnealedSDE(BaseSDE):
    def __init__(
        self,
        source,
        energy,
        beta0: float = 20.0,
        beta1: float = 0.1,
        sigma: float = 1.0,
        max_grad_E_norm: float = False,
        type: str = 'annealed'
    ):
        super().__init__()
        self._source = source
        self._energy = energy
        self.max_grad_E_norm = max_grad_E_norm
        self.type = type

        self.register_buffer(
            "beta1",
            torch.tensor(beta1, dtype=torch.float),
            persistent=False,
        )
        self.register_buffer(
            "beta0",
            torch.tensor(beta0, dtype=torch.float),
            persistent=False,
        )
        self.register_buffer(
            "sigma",
            torch.tensor(sigma, dtype=torch.float),
            persistent=False,
        )

    def U0(self, x: torch.Tensor) -> torch.Tensor:
        # p(x) = C exp(-E(x)), E(x) = - log p(x)
        return - self._source.unnorm_log_prob(x)

    def grad_U0(self, x: torch.Tensor) -> torch.Tensor:
        # p(x) = C exp(-E(x)), ∇ E(x) = - ∇ log p(x)
        return - self._source.score(x)

    def U1(self, x: torch.Tensor) -> torch.Tensor:
        return self._energy.eval(x)
    
    def grad_U1(self, x1):
        grad_E = self._energy(x1)["forces"]

        if self.max_grad_E_norm:
            norm = torch.linalg.vector_norm(grad_E, dim=-1).detach()
            clip_coefficient = torch.clamp(self.max_grad_E_norm / (norm + 1e-6), max=1)
            clip_coefficient = clip_coefficient.unsqueeze(-1)
        else:
            clip_coefficient = torch.ones_like(grad_E)
        
        # print(clip_coefficient.min())
        
        return grad_E * clip_coefficient
    
    def grad_Ut(self, t:torch.Tensor, x:torch.Tensor) -> torch.Tensor:
        if self.type == 'annealed':
            return (1 - t) * self.grad_U0(x) + t * self.grad_U1(x)
        elif self.type == 'langevin':
            return self.grad_U1(x)

    def _beta(self, t: torch.Tensor) -> torch.Tensor:
        return torch.lerp(self.beta0, self.beta1, t)

    def drift(self, t: torch.Tensor, x: torch.Tensor, return_u: bool | bool = False) -> torch.Tensor:
        annealed_drift = self.sigma**2 * self.grad_Ut(t, x)
        
        if return_u:
            return - 0.5 * self._beta(t) * annealed_drift, 0.5 * (x - annealed_drift) / (self.sigma**2 + 1e-8)
        else: 
            return - 0.5 * self._beta(t) * annealed_drift

    def diff(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma * torch.sqrt(self._beta(t))

    def coeff2(self, t):
        bt, b1 = self._beta(t), self.beta1
        return -0.25 * (1-t) * (bt + b1) # = $ -\\frac{1}{2} \int^1_t \\beta(s) ds$

    def sample_posterior(self, t, x0, x1, expand_dim=False):
        bt, b0, b1 = self._beta(t), self.beta0, self.beta1
        coeff1 = -0.25 * t * (bt + b0) # = $ -\\frac{1}{2} \int^t_0 \\beta(s) ds$
        coeff2 = -0.25 * (1-t) * (bt + b1) # = $ -\\frac{1}{2} \int^1_t \\beta(s) ds$
        coeff3 = -0.25 * (b1 + b0)

        mu = torch.exp(coeff1) * (1 - torch.exp(2*coeff2)) / (1 - torch.exp(2*coeff3)) * x0 \
           + torch.exp(coeff2) * (1 - torch.exp(2*coeff1)) / (1 - torch.exp(2*coeff3)) * x1

        var = (1 - torch.exp(2*coeff1)) * (1 - torch.exp(2*coeff2)) / (1 - torch.exp(2*coeff3) + 1e-8)
        var[var < 0] = 0
        std = self.sigma * torch.sqrt(var)

        z = torch.randn_like(mu)
        xt = mu + std * z
        
        score = - torch.exp(coeff2) * (torch.exp(coeff2) * xt - x1) / (self.sigma**2 * (1 - torch.exp(2*coeff2)) + 1e-6)
        # _, score = self.drift(t, xt, return_u=True)
        return xt, score, self.sigma**2 * (1 - torch.exp(2*coeff2))
    
    def sample_base_posterior(self, t, x0, x1):
        return self.sample_posterior(t, x0, x1)
    
    def train(self, boolean):
        pass


## Graph ##
class Graph:
    def __init__(self, n_particles: int = 3, spatial_dim: int | None = None):
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim
        self.projcted_gauss = CenteredParticlesGauss(
            n_particles,
            spatial_dim,
            scale=1,
        )

    def is_freemean(self, x: torch.Tensor):
        return graph_utils.is_freemean(
            x,
            self.n_particles,
            self.spatial_dim
        )

    def randn_like(self, x: torch.Tensor):
        B, D = x.shape
        assert D == self.spatial_dim * self.n_particles

        noise = self.projcted_gauss.sample((B,)).to(x)
        assert noise.shape == x.shape
        return noise


# note: Graph goes before BaseSDE to override rand_like!
class GraphBMSDE(Graph, BrownianMotionSDE):
    """ dX_t = σ A dW_t, where A is the matrix such that y = Ax has zero COM
        note: _pt_gauss_param output intermideate results
    """
    def __init__(self, n_particles: int = 3, spatial_dim: int | None = None, *args, **kwargs):
        Graph.__init__(self, n_particles, spatial_dim)
        BrownianMotionSDE.__init__(self, *args, **kwargs)

    def pt_gauss_param(self, *args):
        raise NotImplementedError("This should never be called!")

    def sample_posterior(self, t, x0, x1):
        xt, score, var = super().sample_posterior(t, x0, x1)
        return xt, score, var


class GraphVESDE(Graph, VESDE):
    """ dX_t = σ A dW_t, where A is the matrix such that y = Ax has zero COM
        note: _pt_gauss_param output intermideate results
    """
    def __init__(self, n_particles: int = 3, spatial_dim: int | None = None, *args, **kwargs):
        Graph.__init__(self, n_particles, spatial_dim)
        VESDE.__init__(self, *args, **kwargs)

    def pt_gauss_param(self, *args):
        raise NotImplementedError("This should never be called!")

    def sample_posterior(self, t, x0, x1):
        xt, score, var = super().sample_posterior(t, x0, x1)
        return xt, score, var


class GraphVPSDE(Graph, VPSDE):
    """ dX_t = σ A dW_t, where A is the matrix such that y = Ax has zero COM
        note: _pt_gauss_param output intermideate results
    """
    def __init__(self, n_particles, spatial_dim, beta0, beta1, sigma, **kwargs):
        Graph.__init__(self, n_particles, spatial_dim)
        VPSDE.__init__(self, beta0, beta1, sigma, **kwargs)

    def pt_gauss_param(self, *args):
        raise NotImplementedError("This should never be called!")

    def sample_posterior(self, t, x0, x1):
        xt, score, var = super().sample_posterior(t, x0, x1)
        return xt, score, var
    
class GraphAnnealedSDE(Graph, AnnealedSDE):
    """ dX_t = σ A dW_t, where A is the matrix such that y = Ax has zero COM
        note: _pt_gauss_param output intermideate results
    """
    def __init__(self, n_particles, spatial_dim,
        source,
        energy,
        beta0: float = 20.0,
        beta1: float = 0.1,
        sigma: float = 1.0,
        max_grad_E_norm: float = False,
        type: str = 'annealed'
        , **kwargs):
        Graph.__init__(self, n_particles, spatial_dim)
        AnnealedSDE.__init__(self, source, energy, beta0, beta1, sigma, max_grad_E_norm, type, **kwargs)

    def pt_gauss_param(self, *args):
        raise NotImplementedError("This should never be called!")

    def sample_posterior(self, t, x0, x1):
        xt, score, var = super().sample_posterior(t, x0, x1)
        return xt, score, var


## Chemistry ##
class Chem:
    def randn_like(self, graph_state: Batch) -> torch.Tensor:
        return graph_utils.subtract_com_batch(
            torch.randn_like(graph_state["positions"]),
            graph_state["batch"],
        )

    def propagate(self, graph_state: Batch, d_position: torch.Tensor) -> Batch:
        # NOTE(ghliu): we do NOT create new graph_state (Batch)!!
        graph_state["positions"] = graph_state["positions"] + d_position
        graph_state["positions"] = graph_utils.subtract_com_batch(
            graph_state["positions"], graph_state["batch"]
        )
        return graph_state


# NOTE: NEED TO BE MODIFIED
# note: Chem goes before BaseSDE to override rand_like & propagate!
# NOTE: rand_like & drift output positions not Batch!
class ChemVESDE(Chem, VESDE):
    """ dX_t = σ A dW_t, where A is the matrix such that y = Ax has zero COM
        note: _pt_gauss_param output intermideate results
    """
    def __init__(self, *args, **kwargs):
        Chem.__init__(self)
        VESDE.__init__(self, *args, **kwargs)

    def drift(self, t, graph_state: Batch) -> torch.Tensor:
        return torch.zeros_like(graph_state["positions"])

    def pt_gauss_param(self, *args):
        raise NotImplementedError("This should never be called!")

    def sample_posterior(
        self,
        t: torch.Tensor,
        graph_state0: Batch,
        graph_state1: Batch,
    ) -> Batch:
        assert graph_utils.is_zcom_graph(graph_state0)
        assert graph_utils.is_zcom_graph(graph_state1)

        (B, D), N = graph_state0["positions"].shape, t.shape

        x0 = graph_state0["positions"]
        x1 = graph_state1["positions"]
        tt = t[graph_state0["batch"], None]
        assert x0.shape == x1.shape == (B, D) and tt.shape == (B, 1)

        z = self.randn_like(graph_state0)
        xt = super().sample_posterior(tt, x0, x1, z=z)
        assert xt.shape == (B, D)

        graph_state_t = graph_utils.create_new_graph(
            graph_state0, xt, subtract_com=True
        )
        return graph_state_t

    def cond_score(
        self,
        graph_state0: Batch,
        t: torch.Tensor,
        graph_state_t: Batch,
    ) -> torch.Tensor:
        assert graph_utils.is_zcom_graph(graph_state0)
        assert graph_utils.is_zcom_graph(graph_state_t)

        (B, D), N = graph_state0["positions"].shape, t.shape

        x0 = graph_state0["positions"]
        xt = graph_state_t["positions"]
        tt = t[graph_state0["batch"], None]
        assert x0.shape == xt.shape == (B, D) and tt.shape == (B, 1)

        score = super().cond_score(x0, tt, xt)
        assert score.shape == (B, D)
        # TODO(ghliu) subtract zero?
        return score


class ControlledSDE(BaseSDE):
    """ dX_t = ( b(t,x) + g(t)^2 u(t,x) )(t, X_t) dt + g(t) dW_t"""
    def __init__(
        self,
        ref_sde: BaseSDE,
        u: torch.nn.Module,
        param_type: str
    ):
        super().__init__()
        assert param_type in ['adjoint', 'control']
        self.ref_sde = ref_sde
        self.u = u
        self.param_type = param_type

    def sample_base_posterior(self, t, x0, x1):
        # p^{base}_t(x | x0, x1)
        return self.ref_sde.sample_posterior(t, x0, x1)

    def randn_like(self, x):
        return self.ref_sde.randn_like(x)

    def propagate(self, x, dx):
        return self.ref_sde.propagate(x, dx)

    def diff(self, t):
        return self.ref_sde.diff(t)

    def drift(self, t, x, return_u=False):
        u = self.u(t, x)
        sigma = self.diff(t)
        if return_u:
            if self.param_type == 'control':
                return self.ref_sde.drift(t, x) + sigma * u, u
            elif self.param_type == 'adjoint':
                return self.ref_sde.drift(t, x) + (sigma**2) * u, sigma * u
        else:
            if self.param_type == 'control':
                return self.ref_sde.drift(t, x) + sigma * u
            elif self.param_type == 'adjoint':
                return self.ref_sde.drift(t, x) + (sigma**2) * u


@torch.no_grad()
def sdeint(
    sde: BaseSDE,
    state0: torch.Tensor | Batch,
    timesteps: torch.Tensor,
    zero_last_step_noise: bool = False,
    only_boundary: bool = False,
) -> List[torch.Tensor | Batch]:
    if isinstance(state0, Batch):
        assert only_boundary, "Does not support return trajectories for Batch X_0!"

    T = len(timesteps)
    assert len(timesteps) > 1

    # note: always use EMA.
    sde.train(False)

    state = state0.clone()

    states = [state0,]
    for i in tqdm(range(T - 1), desc="SDE integration", unit="step"):
        t = timesteps[i]
        dt = timesteps[i + 1] - t

        drift = sde.drift(t, state) * dt
        diffusion = sde.diff(t) * dt.sqrt() * sde.randn_like(state)

        zero_noise = zero_last_step_noise and i == (T - 2)
        d_state = drift if zero_noise else (drift + diffusion)

        # euler maruyama step
        state = sde.propagate(state, d_state)
        # if isinstance(sde.ref_sde, Graph): assert sde.ref_sde.is_freemean(state)
        # if isinstance(sde.ref_sde, Chem): assert graph_utils.is_zcom_graph(state)
        states.append(state)


    if only_boundary:
        return states[0], states[-1]
    return states



@torch.no_grad()
def loss_sdeint(
    sde: BaseSDE,
    state0: torch.Tensor | Batch,
    timesteps: torch.Tensor,
    zero_last_step_noise: bool = False,
    only_boundary: bool = False,
) -> List[torch.Tensor | Batch]:
    if isinstance(state0, Batch):
        assert only_boundary, "Does not support return trajectories for Batch X_0!"

    T = len(timesteps)
    assert len(timesteps) > 1

    # note: always use EMA.
    sde.train(False)

    state = state0.clone()

    states = [state0,]
    weight = 0
    for i in tqdm(range(T - 1), desc="SDE integration", unit="step"):
        t = timesteps[i]
        dt = timesteps[i + 1] - t

        drift, u = sde.drift(t, state, return_u=True) 
        z = dt.sqrt() * sde.randn_like(state)

        zero_noise = zero_last_step_noise and i == (T - 2)
        d_state = drift * dt if zero_noise else (drift * dt + sde.diff(t) * z)

        # euler maruyama step
        state = sde.propagate(state, d_state).clone()
        states.append(state)

        weight += - 0.5 * (u**2).sum(1) * dt - (u * z).sum(1)
    if only_boundary:
        return (states[0], states[-1]), weight

    return states, weight



