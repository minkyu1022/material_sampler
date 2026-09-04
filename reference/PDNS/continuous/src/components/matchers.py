import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from mace.tools.torch_geometric import Batch
from src.components.buffer import BatchBuffer, chem_collate
from src.components.sdes import BaseSDE, VPSDE, loss_sdeint
from ipdb import set_trace as debug
from src.components.term_cost import TermCost
from src.components.scheduler import BaseScheduler
import matplotlib.pyplot as plt


class ScoreMatcher:
    def __init__(
        self,
        source,
        buffer_size: int | None=None,
        duplicates: int | None = None,
        term_cost: TermCost | None = None,
        gamma: BaseScheduler | None = None,
        alpha: float | None = None,
        iws: bool | None = None,
        cumulate: bool | None = None,
        **kwargs
        ):
        self.source = source
        self.buffer = BatchBuffer(buffer_size)
        self.duplicates = duplicates
        self.term_cost = term_cost
        self.gamma = gamma
        self.alpha = alpha
        self.beta = 1
        self.iws = iws
        self.cumulate = cumulate
        self.count = 0

    def populate_buffer(
            self,
            x0: torch.Tensor,
            sde: BaseSDE,
            timesteps: torch.Tensor,
            zero_last_step_noise: bool = False,
    ):
        (x0, x1), log_rnd = loss_sdeint(
            sde,
            x0,
            timesteps,
            zero_last_step_noise=zero_last_step_noise,
            only_boundary=True,
        )
        # if self.cumulate:
        #     log_rnd -= (1-self.beta) * self.term_cost(x1) / self.alpha
        # else:
        #     log_rnd -= self.term_cost(x1) / self.alpha
        
        assert len(log_rnd) == len(x1)

        self.buffer.add({
            "x0": x0.to("cpu"),
            "x1": x1.to("cpu"),
            "log_rnd": log_rnd.to("cpu"),
            "term_cost": (self.term_cost(x1) / self.alpha).to("cpu")
        })

    def sample_t(self, x):
        (B, D) = x.shape
        return torch.rand(B, 1)

    def build_dataloader(self, batch_size, stage, collate_fn=None, adapt_scheduler=True) -> DataLoader:
        # build dataset
        dataset = self.buffer.build_dataset(self.duplicates)

        if adapt_scheduler:
            dataset.total_data = self.gamma.update(dataset.total_data, stage, self.cumulate)
        else:
            dataset.total_data = self.gamma.apply(dataset.total_data, stage, self.cumulate)
        
        if self.iws:
            # compute weights
            weight = dataset.total_data['weight']
            
            # check if there is an NAN values, if so get rid of it in data:
            mask = 1*(torch.isnan(weight)) + 1*(weight>1e8) + 1*(weight<0) == 0
            for key, item in dataset.total_data.items():
                dataset.total_data[key] = item[mask]
            weight = dataset.total_data['weight']
            
            # normalize weight and do importance weighted sampling
            normalized_weight = weight / sum(weight)
            normalized_weight[normalized_weight<0] = 0
            indices = torch.multinomial(normalized_weight, len(weight), replacement=True)
            for key, item in dataset.total_data.items():
                dataset.total_data[key] = item[indices]
            
            # set all weights to one
            dataset.total_data['weight'] = torch.ones(len(weight))
            
        # recompute 
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,), dataset.total_data, self.gamma()
    
    def prepare_target(self, data, sde, device):
        x0, x1, weight = self.source.sample([len(data['x1']),]).to(device), \
                        data['x1'].to(device), \
                        data['weight'].to(device)
        self.count += 1

        # get random t
        t = (1 - 1e-3) * torch.rand((len(x0), 1), device=device)
        xt, score, var = sde.sample_base_posterior(t, x0, x1)
        diff_term = sde.diff(t)
        total_diff_term = var.sqrt()
        return (t, xt), diff_term, total_diff_term, score, weight
    
    def clean_buffer(self):
        self.buffer.clean()



