import torch
import torch.distributions as dist
import torch.nn.functional as F
from src.energies.dist_utils import Distribution

class StudentTMixtureModel(Distribution):
    def __init__(self, num_components, dim, log_Z=0., can_sample=True, sample_bounds=None, device='cuda:0'):
        super().__init__(dim=dim, log_norm_const=log_Z)

        # Parameters
        self.num_components = num_components
        self.dim = dim
        self.log_Z = log_Z
        self.can_sample = can_sample
        
        # Set random seed for reproducibility
        torch.manual_seed(0)

        # Parameters for mixture components
        min_mean_val = -10
        max_mean_val = 10
        min_val_mixture_weight = 0.3
        max_val_mixture_weight = 0.7
        degree_of_freedoms = 2
        
        # Mixture components' locations (means), scales (std), and degrees of freedom
        locs = torch.FloatTensor(self.num_components, self.dim).uniform_(min_mean_val, max_mean_val).to(device).float()
        dofs = torch.full((self.num_components, self.dim), degree_of_freedoms).to(device).float()
        scales = torch.ones((self.num_components, self.dim)).to(device).float()

        # Create the Student-T distribution components
        self.component_dist = dist.Independent(dist.StudentT(df=dofs, loc=locs, scale=scales), 1)

        # Create the mixture weights (either uniform or randomly generated)
        logits = torch.ones(self.num_components) / self.num_components
        mixture_weights = dist.Categorical(logits=logits.to(device))
        
        
        # Mixture model distribution
        self.mixture_distribution = dist.MixtureSameFamily(mixture_weights, self.component_dist)

    def sample(self, sample_shape):
        return self.mixture_distribution.sample(sample_shape)

    def log_prob(self, x):
        log_prob = self.mixture_distribution.log_prob(x)
        return log_prob
    
    def unnorm_log_prob(self, x):
        return self.log_prob(x)

    def entropy(self, samples):
        # Calculate the entropy of the mixture model
        log_probs = self.mixture_distribution.component_log_probs(samples)
        idx = torch.argmax(log_probs, dim=1)
        unique_elements, counts = idx.unique(return_counts=True)
        mode_dist = counts.float() / samples.shape[0]
        entropy = -(mode_dist * (mode_dist.log() / torch.log(torch.tensor(self.num_components, dtype=torch.float32)))).sum()
        return entropy