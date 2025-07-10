import torch, time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy


class AbsLoss(object):
    r"""An abstract class for loss functions. 
    """

    def __init__(self):
        self.record = []
        self.bs = []

    def compute_loss(self, pred, gt):
        r"""Calculate the loss.
        
        Args:
            pred (torch.Tensor): The prediction tensor.
            gt (torch.Tensor): The ground-truth tensor.

        Return:
            torch.Tensor: The loss.
        """
        pass

    def update_loss(self, pred, gt):
        loss = self.compute_loss(pred, gt)
        self.record.append(loss.item())
        self.bs.append(pred.size()[0])
        return loss

    def average_loss(self):
        record = np.array(self.record)
        bs = np.array(self.bs)
        return (record * bs).sum() / bs.sum()

    def reinit(self):
        self.record = []
        self.bs = []


class CELoss(AbsLoss):
    r"""The cross-entropy loss function.
    """

    def __init__(self):
        super(CELoss, self).__init__()

        self.loss_fn = nn.CrossEntropyLoss()

    def compute_loss(self, pred, gt):
        r"""
        """
        loss = self.loss_fn(pred, gt)
        return loss


class AbsMetric(object):
    r"""An abstract class for the performance metrics of a task. 

    Attributes:
        record (list): A list of the metric scores in every iteration.
        bs (list): A list of the number of data in every iteration.
    """

    def __init__(self):
        self.record = []
        self.bs = []

    @property
    def update_fun(self, pred, gt):
        r"""Calculate the metric scores in every iteration and update :attr:`record`.

        Args:
            pred (torch.Tensor): The prediction tensor.
            gt (torch.Tensor): The ground-truth tensor.
        """
        pass

    @property
    def score_fun(self):
        r"""Calculate the final score (when an epoch ends).

        Return:
            list: A list of metric scores.
        """
        pass

    def reinit(self):
        r"""Reset :attr:`record` and :attr:`bs` (when an epoch ends).
        """
        self.record = []
        self.bs = []


# accuracy
class AccMetric(AbsMetric):
    r"""Calculate the accuracy.
    """

    def __init__(self):
        super(AccMetric, self).__init__()

    def update_fun(self, pred, gt):
        r"""
        """
        pred = F.softmax(pred, dim=-1).max(-1)[1]
        self.record.append(gt.eq(pred).sum().item())
        self.bs.append(pred.size()[0])

    def score_fun(self):
        r"""
        """
        return (sum(self.record) / sum(self.bs))


# L1 Error
class L1Metric(AbsMetric):
    r"""Calculate the Mean Absolute Error (MAE).
    """

    def __init__(self):
        super(L1Metric, self).__init__()

    def update_fun(self, pred, gt):
        r"""
        """
        abs_err = torch.abs(pred - gt)
        self.record.append(abs_err.item())
        self.bs.append(pred.size()[0])

    def score_fun(self):
        r"""
        """
        records = np.array(self.record)
        batch_size = np.array(self.bs)
        return (records * batch_size).sum() / (sum(batch_size))


# task_name = ['Art', 'Clipart', 'Product', 'Real_World']


def prepare_dataloaders(dataloaders, task_name):
    loader = {}
    batch_num = []
    for task in task_name:
        loader[task] = [dataloaders[task], iter(dataloaders[task])]
        batch_num.append(len(dataloaders[task]))
    return loader, batch_num


def process_data(loader, device):
    try:
        data, label = next(loader[1])
    except:
        loader[1] = iter(loader[0])
        data, label = next(loader[1])
    data = data.to(device, non_blocking=True)
    label = label.to(device, non_blocking=True)
    return data, label


class GradEstimator(object):
    """Estimate gradient using zero order approximation.
    """

    def __init__(self, model, eps=1e-2):
        self.model = deepcopy(model)
        self.eps = eps
        self.loss_fn = nn.CrossEntropyLoss()

    def perturb_parameters(self, random_seed, scaling_factor=1):
        torch.manual_seed(random_seed)
        for n, p in self.model.named_parameters():
            if n.startswith("out_layers"):
                continue
            z = torch.normal(mean=0,
                             std=1.0,
                             size=p.data.size(),
                             device=p.data.device,
                             dtype=p.data.dtype)
            p.data = p.data + scaling_factor * z * self.eps

    def forward(self, data, task):
        random_seed = np.random.randint(1000000000)
        est_input, est_gt = data
        with torch.no_grad():
            # first function evaluation f(x+\delta)
            self.perturb_parameters(random_seed, scaling_factor=1)
            self.model.eval()
            out = self.model(est_input, task)
            loss1 = self.loss_fn(out, est_gt)

            # second function evaluation f(x-\delta)
            self.perturb_parameters(random_seed, scaling_factor=-2)
            self.model.eval()
            out = self.model(est_input, task)
            loss2 = self.loss_fn(out, est_gt)

        projected_grad = (loss1 - loss2) / (2 * self.eps)
        model_dtype = next(self.model.parameters()).dtype
        projected_grad = projected_grad.to(model_dtype)

        # reset model back to its parameters at start of step
        self.perturb_parameters(random_seed, scaling_factor=1)
        torch.manual_seed(random_seed)
        zeroth_grad = {}
        for n, p in self.model.named_parameters():
            if n.startswith("out_layers"):
                continue
            temp_grad = projected_grad * torch.normal(mean=0,
                                                      std=1.0,
                                                      size=p.data.size(),
                                                      device=p.data.device,
                                                      dtype=p.data.dtype)
            zeroth_grad[n] = temp_grad
        return zeroth_grad
