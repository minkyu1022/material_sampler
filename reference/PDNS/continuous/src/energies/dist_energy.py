from ast import Dict
import torch

from src.energies.dist_utils import Distribution
from src.energies.base_energy_function import BaseEnergy


class DistEnergy(BaseEnergy):
    """ An energy function given a distribution
    """
    def __init__(self, dist: Distribution, device="cpu"):
        super().__init__(name=dist.name, dim=dist.dim)
        self.dist = dist
        self.dist.to(device)
        self.device = device
        self.is_molecule = False

    @torch.no_grad()
    def sample(self, shape):
        # note: this function should be called only in eval.
        assert hasattr(self.dist, "sample"), f"Energy {self.name} does not support sampling!"
        return self.dist.sample(shape)

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        # p(x) = C exp(-E(x)) --> E(x) = - log p(x)
        return - self.dist.unnorm_log_prob(x)

    def grad_E(self, x):
        # return analytic score if implemented,
        # otherwise fall back to default autograd
        if hasattr(self.dist, "score"):
            return - self.dist.score(x)
        else:
            return super().grad_E(x)
        
    def unnormalize(self, x):
        return x

    def normalize(self, x):
        return x

