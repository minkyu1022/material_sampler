from typing import Optional
import torch
from mace.calculators.foundations_models import mace_off
from mace.tools import utils

from src.utils.chem_utils import bond_structure_regularizer



class MaceEnergy(torch.nn.Module):
    def __init__(
        self,
        tau=1e-3,
        alpha=1e3,
        device="cpu",
        default_dtype="float64",
        default_regularize: bool = False, # NOTE(ghliu)
    ):
        super().__init__()
        self.calc = mace_off(device=device, default_dtype=default_dtype)
        model = self.calc.models[0]
        model = model.to(device)
        self.model = model
        self.device = device
        self.tau = tau
        self.alpha = alpha
        self.r_max = model.r_max
        self.atomic_numbers = model.atomic_numbers
        self.default_regularize = default_regularize
        self.z_table = utils.AtomicNumberTable([int(z) for z in self.atomic_numbers])
        self.name = "mace"

    def bond_regularizer(self, batch):
        #batch["positions"].requires_grad = True
        energy_reg = bond_structure_regularizer(
            batch["positions"],
            batch["edge_attrs"][:, 0].unsqueeze(-1),
            batch["edge_attrs"][:, 1].unsqueeze(-1),
            batch["edge_index"],
            batch["ptr"],
            alpha=self.alpha,
        )
        grad_outputs = [torch.ones_like(energy_reg)]
        gradient = torch.autograd.grad(
            outputs=[energy_reg],  # [n_graphs, ]
            inputs=[batch["positions"]],  # [n_nodes, 3]
            grad_outputs=grad_outputs,
            retain_graph=False,  # Make sure the graph is not destroyed during training
            create_graph=False,  # Create graph for second derivative
            allow_unused=True,  # For complete dissociation turn to true
        )[0]
        return energy_reg, -gradient

    def __call__(self, batch, regularize: Optional[bool] = None):
        if regularize is None:
            regularize = self.default_regularize

        output_dict = self.model(batch)

        output_dict["forces"] = (output_dict["forces"].detach()) / self.tau

        if regularize:
            reg_energy, reg_force = self.bond_regularizer(batch)
            output_dict["reg_forces"] = reg_force.detach()
            output_dict["reg_energy"] = reg_energy.detach()
        else:
            output_dict["reg_forces"] = torch.zeros_like(output_dict["forces"])
            output_dict["reg_energy"] = torch.zeros_like(output_dict["energy"])

        return output_dict
