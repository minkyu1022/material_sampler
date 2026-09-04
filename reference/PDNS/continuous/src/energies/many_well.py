import math
import torch
import matplotlib.pyplot as plt
from typing import List
import itertools
import wandb
from src.energies.dist_utils import Distribution

class ManyWellNew(Distribution):
    def __init__(
        self,
        dim: int = 5,
        m: int = 5,
        delta: float = 4,
        can_sample: bool = False,
        sample_bounds=None,
    ):
        self.d = dim
        self.m = m
        self.delta = delta
        self._plot_bound = 3.0

        super().__init__(dim=dim, log_norm_const=self.log_Z)

    def log_prob(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)

        prefix = x[:, :self.m]
        suffix = x[:, self.m:]
        k = ((prefix**2 - self.delta) ** 2).sum(dim=1)
        k2 = 0.5 * (suffix**2).sum(dim=1)

        log_probs = -k - k2
        return log_probs.squeeze()
    
    def unnorm_log_prob(self, x):
        return self.log_prob(x)

    @property
    def log_Z(self):
        l, r = -100, 100
        s = 100000000

        pt = torch.rand(s) * (r - l) + l
        fst = torch.log(torch.sum(torch.exp(-((pt**2 - self.delta) ** 2))) * ((r - l) / s))
        self.logZ_1d = fst.item()

        pt2 = torch.rand(s) * (r - l) + l
        snd = torch.log(torch.sum(torch.exp(-0.5 * pt2**2)) * ((r - l) / s))

        return fst * self.m + snd * (self.d - self.m)

    def sample(self, sample_shape: torch.Size) -> torch.Tensor:
        REJECTION_SCALE = 6
        n_samples = math.prod(sample_shape)
        n_dw, n_gauss = self.m, self.d - self.m

        def double_well_1d_log_density(xs, shift, separation):
            return -((xs - shift) ** 2 - separation) ** 2 - self.logZ_1d

        def get_proposal(shift, separation):
            loc = shift + math.sqrt(separation) * torch.tensor([[-1.0], [1.0]])
            scale = 1 / math.sqrt(separation) * torch.tensor([[1.0], [1.0]])
            mix = torch.distributions.Categorical(torch.tensor([0.5, 0.5]))
            comp = torch.distributions.Independent(
                torch.distributions.Normal(loc, scale), 1
            )
            return torch.distributions.MixtureSameFamily(mix, comp)

        def rejection_sampling(shape, proposal, target_pdf, scaling):
            total_samples = math.prod(shape)
            accepted = []

            while len(accepted) < total_samples:
                xs = proposal.sample((total_samples * 10,))
                u = torch.rand(xs.shape[0])
                log_p = torch.log(target_pdf(xs).squeeze())
                log_q = proposal.log_prob(xs)
                accept = u < torch.exp(log_p - log_q) / scaling
                accepted.extend(xs[accept].tolist())

            accepted_tensor = torch.tensor(accepted[:total_samples]).reshape(shape)
            return accepted_tensor

        def sample_1d_double_well(shape, shift, separation):
            proposal = get_proposal(shift, separation)
            target_pdf = lambda x: torch.exp(double_well_1d_log_density(x, shift, separation))
            return rejection_sampling(shape, proposal, target_pdf, REJECTION_SCALE)

        # Sample double well for each 1D axis
        dw_samples = [
            sample_1d_double_well(sample_shape, shift=0.0, separation=self.delta)
            for _ in range(n_dw)
        ]
        dw_samples = torch.stack(dw_samples, dim=-1)

        # Gaussian tail
        gauss_samples = torch.randn(*sample_shape, n_gauss)

        return torch.cat([dw_samples, gauss_samples], dim=-1)