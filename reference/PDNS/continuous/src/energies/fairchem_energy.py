import torch

#SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
from fairchem.experimental.xiangfu.md.utils.mt_calc import Predictor

from src.utils.chem_utils import bond_structure_regularizer

class FairChemEnergy(torch.nn.Module):
    def __init__(
        self,
        model_ckpt,
        dataset_name="spice",
        key_mapping={"spice_energy": "energy", "spice_forces": "forces"},
        tau=1e-3, alpha=1e3, device="cpu",
    ):
        super().__init__()
        predictor = Predictor(model_ckpt, dataset_name)
        self.predictor = predictor
        self.predictor.model.backbone.use_pbc = False
        self.predictor.model.backbone.use_pbc_single = True
        self.predictor.model.to(device)
        self.predictor.model.device = device
        self.device = device
        self.tau = tau
        self.alpha = alpha
        self.r_max = predictor.model.backbone.cutoff
        self.atomic_numbers = torch.arange(100)
        self.key_mapping = key_mapping

        self.name = "eSEN"


    def bond_regularizer(self, batch):
        energy_reg = bond_structure_regularizer(
            batch["positions"],
            batch["edge_attrs"][:, 0].unsqueeze(-1), # bond limits
            batch["edge_attrs"][:, 1].unsqueeze(-1), # bond types
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

    def __call__(self, batch, regularize=True):

        # rename/add required input fields for fairchem model
        batch.natoms = batch.ptr[1:] - batch.ptr[:-1]
        batch.atomic_numbers = batch.node_attrs.argmax(dim=-1)

        # TODO maybe turn off otf graph
        batch.cell = batch.cell.view(-1,3,3) + torch.eye(3).to(batch.cell.device).unsqueeze(0) * 1e3
        batch.pos = batch.positions.float() # wrap?
        # note that our model has otf graph. edge_index here is not used.
        preds = self.predictor.predict(batch)
        output_dict = {}
        for k, v in preds.items():
            output_dict[self.key_mapping[k]] = v

        if regularize:
            reg_energy, reg_force = self.bond_regularizer(batch)
            output_dict["reg_forces"] = (reg_force.detach()) # / self.tau
            output_dict["reg_energy"] = reg_energy.detach()
        else:
            output_dict["reg_forces"] = torch.zeros_like(output_dict["forces"])
            output_dict["reg_energy"] = torch.zeros_like(output_dict["energy"])

        output_dict["forces"] = (output_dict["forces"].detach()) / self.tau
        return output_dict

