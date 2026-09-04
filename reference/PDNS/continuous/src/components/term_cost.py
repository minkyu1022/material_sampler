import torch

class TermCost:
    def __init__(self, energy, sigma, clip_term_norm=False):
        self.energy = energy
        self.sigma = sigma
        self.clip_term_norm = clip_term_norm

    def unnormalized_prior(self, x):
        return - (x**2).sum(1) / (2 * self.sigma**2)

    def __call__(self, x1):
        term_cost = self.unnormalized_prior(x1) + self.energy.eval(x1).squeeze()
        if self.clip_term_norm:
            term_cost = torch.clamp(term_cost, min=-self.clip_term_norm, max=self.clip_term_norm)
        return term_cost