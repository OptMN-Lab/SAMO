import numpy as np
import torch
from torch_geometric.utils import remove_self_loops
import torch.nn.functional as F

from copy import deepcopy


class MyTransform(object):

    def __init__(self, target: list = None):
        if target is None:
            target = torch.tensor([0, 1, 2, 3, 5, 6, 12, 13, 14, 15,
                                   11])  # removing 4
        else:
            target = torch.tensor(target)
        self.target = target

    def __call__(self, data):
        # Specify target.
        data.y = data.y[:, self.target]
        return data


class Complete(object):

    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data


qm9_target_dict = {
    0: "mu",
    1: "alpha",
    2: "homo",
    3: "lumo",
    5: "r2",
    6: "zpve",
    7: "U0",
    8: "U",
    9: "H",
    10: "G",
    11: "Cv",
}

# for \Delta_m calculations
# -------------------------
# DimeNet uses the atomization energy for targets U0, U, H, and G.
target_idx = [0, 1, 2, 3, 5, 6, 12, 13, 14, 15, 11]

# Report meV instead of eV.
multiply_indx = [2, 3, 5, 6, 7, 8, 9]

n_tasks = len(target_idx)

# stl results
BASE = np.array([
    0.0671,
    0.1814,
    60.576,
    53.915,
    0.5027,
    4.539,
    58.838,
    64.244,
    63.852,
    66.223,
    0.07212,
])
#BASE = np.array(
#    [
#        0.0671,
#        0.1814,
#        0.060576,
#        0.053915,
#        0.5027,
#        0.004539,
#        0.058838,
#        0.064244,
#        0.063852,
#        0.066223,
#        0.07212,
#    ]
#)

SIGN = np.array([0] * n_tasks)
KK = np.ones(n_tasks) * -1


def delta_fn(a):
    return (KK**SIGN * (a - BASE) / BASE).mean() * 100.0  # *100 for percentage


class GradEstimator(object):
    """Estimate gradient using zero order approximation.
    """

    def __init__(self, model, eps=1e-2):
        self.model = deepcopy(model)
        self.eps = eps

    def perturb_parameters(self, random_seed, scaling_factor=1):
        torch.manual_seed(random_seed)
        for n, p in self.model.named_parameters():
            if n.startswith("head"):
                continue
            z = torch.normal(mean=0,
                             std=1.0,
                             size=p.data.size(),
                             device=p.data.device,
                             dtype=p.data.dtype)
            p.data = p.data + scaling_factor * z * self.eps

    def forward(self, inputs, task):
        random_seed = np.random.randint(1000000000)

        with torch.no_grad():
            # first function evaluation f(x+\delta)
            self.perturb_parameters(random_seed, scaling_factor=1)
            self.model.eval()
            out, _ = self.model(inputs, return_representation=True)
            loss1 = F.mse_loss(out, inputs.y, reduction="none").mean(0)[task]

            # second function evaluation f(x-\delta)
            self.perturb_parameters(random_seed, scaling_factor=-2)
            self.model.eval()
            out, _ = self.model(inputs, return_representation=True)
            loss2 = F.mse_loss(out, inputs.y, reduction="none").mean(0)[task]

        projected_grad = (loss1 - loss2) / (2 * self.eps)
        model_dtype = next(self.model.parameters()).dtype
        projected_grad = projected_grad.to(model_dtype)

        # reset model back to its parameters at start of step
        self.perturb_parameters(random_seed, scaling_factor=1)
        torch.manual_seed(random_seed)
        zeroth_grad = {}
        for n, p in self.model.named_parameters():
            if n.startswith("head"):
                continue
            temp_grad = projected_grad * torch.normal(mean=0,
                                                      std=1.0,
                                                      size=p.data.size(),
                                                      device=p.data.device,
                                                      dtype=p.data.dtype)
            zeroth_grad[n] = temp_grad
        return zeroth_grad
