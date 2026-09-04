import torch
import torch.nn as nn
from torch.optim import Adam

class BaseScheduler:
    def __init__(self, epsilon, fix_gamma, device):
        self.beta = 1
        self.epsilon = epsilon
        self.fix_gamma = fix_gamma
        self.device = device
        self._gamma = nn.Parameter(torch.zeros(1, device=device)-3, requires_grad=True)
        self.stage = 0

    def __call__(self):
        if self.fix_gamma:
            return self.fix_gamma
        elif self.stage == 0:
            return 0
        else:
            return torch.sigmoid(self._gamma).detach().item()

    def update(self, total_data, stage, cumulate):
        self.stage = stage
        
        if stage == 0:
            gamma = 0.0
            loss = 0.0
        
        elif self.fix_gamma:
            assert self.fix_gamma > 0
            self.beta = self.beta * (1 - self.fix_gamma)
        
        else:
            lr = 1.0
            while True:
                log_rnd = (total_data['log_rnd'] - total_data['term_cost']).to(self.device)
                N = len(log_rnd)
                f = lambda gamma: (self.epsilon + (N*torch.softmax(log_rnd*torch.sigmoid(gamma), dim=0)+1e-8).log().mean())**2

                opt_f = torch.optim.LBFGS(
                                        [self._gamma],
                                        lr=lr,                # step size for the line search
                                        max_iter=25,           # max closure evaluations per .step()
                                        history_size=10,
                                        line_search_fn="strong_wolfe",
                                    )
                def closure():
                    opt_f.zero_grad()
                    loss = f(self._gamma)
                    loss.backward()
                    return loss

                for it in range(30):           # "outer" iterations
                    loss = opt_f.step(closure)
                if torch.isnan(loss):
                    print('gamma resulted NAN!!')
                    print(self._gamma)
                    self._gamma = nn.Parameter(torch.zeros(1, device=self.device)-3, requires_grad=True)
                    lr *= 0.5
                else:
                    break

            
            gamma = torch.sigmoid(self._gamma)
            self.beta = self.beta * (1 - gamma)

        # apply
        return self.apply(total_data, stage, cumulate)

    def apply(self, total_data, stage, cumulate):
        if stage == 0:
            log_rnd = torch.zeros_like(total_data['log_rnd'])
        
        elif self.fix_gamma:
            log_rnd = total_data['log_rnd']
            term_cost = total_data['term_cost']
            
            if cumulate:
                log_rnd -= (1-self.beta).cpu().detach() * term_cost
            else:
                log_rnd -= term_cost
                log_rnd = self.fix_gamma * log_rnd
        else:
            log_rnd = total_data['log_rnd']
            term_cost = total_data['term_cost']
            
            if cumulate:
                log_rnd -= (1-self.beta).cpu().detach() * term_cost
            else:
                log_rnd -= term_cost
                log_rnd = torch.sigmoid(self._gamma).cpu().detach() * log_rnd
            
        total_data['weight'] = len(log_rnd) * torch.softmax(log_rnd, dim=0)
        return total_data

