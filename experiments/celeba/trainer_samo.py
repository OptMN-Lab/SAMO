import os
from argparse import ArgumentParser

import numpy as np
import time
import torch.utils
import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiments.celeba.data import CelebaDataset
from experiments.celeba.models import Network
from experiments.utils import (
    common_parser,
    extract_weight_method_parameters_from_args,
    get_device,
    set_logger,
    set_seed,
    str2bool,
)
from experiments.celeba.utils import enable_running_stats, disable_running_stats
from experiments.celeba.utils import GradEstimator
from methods.weight_methods import WeightMethods
import ipdb
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import defaultdict


class CelebaMetrics():
    """
    CelebA metric accumulator.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def incr(self, y_preds, ys):
        # y_preds: [ y_pred (batch, 1) ] x 40
        # ys     : [ y_pred (batch, 1) ] x 40
        y_preds = torch.stack(y_preds).detach()  # (40, batch, 1)
        ys = torch.stack(ys).detach()  # (40, batch, 1)
        y_preds = y_preds.gt(0.5).float()
        self.tp += (y_preds * ys).sum([1, 2])  # (40,)
        self.fp += (y_preds * (1 - ys)).sum([1, 2])
        self.fn += ((1 - y_preds) * ys).sum([1, 2])

    def result(self):
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1.cpu().numpy()


def main(path, lr, bs, args, device):
    # we only train for specific task
    model = Network().to(device)

    train_set = CelebaDataset(data_dir=path, split='train')
    val_set = CelebaDataset(data_dir=path, split='val')
    test_set = CelebaDataset(data_dir=path, split='test')

    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=bs,
                                               shuffle=True,
                                               num_workers=2)
    val_loader = torch.utils.data.DataLoader(dataset=val_set,
                                             batch_size=bs,
                                             shuffle=False,
                                             num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=bs,
                                              shuffle=False,
                                              num_workers=2)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epochs = args.n_epochs

    n_tasks = 40
    metrics = np.zeros([epochs, 40], dtype=np.float32)  # test_f1
    metric = CelebaMetrics()
    loss_fn = torch.nn.BCELoss()

    # weight method
    weight_methods_parameters = extract_weight_method_parameters_from_args(
        args)
    weight_method = WeightMethods(args.method,
                                  n_tasks=40,
                                  device=device,
                                  **weight_methods_parameters[args.method])

    best_val_f1 = 0.0
    best_epoch = None

    for epoch in range(epochs):
        # training
        model.train()
        t0 = time.time()
        for x, y in train_loader:
            optimizer.zero_grad()
            x = x.to(device)
            y = [y_.to(device) for y_ in y]
            enable_running_stats(model)
            y_ = model(x)
            losses = torch.stack([
                loss_fn(y_task_pred, y_task)
                for (y_task_pred, y_task) in zip(y_, y)
            ])

            # Get the average gradient
            losses.mean().backward()

            # Get the estimated gradient
            zeroth_grads = {}
            for task in range(n_tasks):
                zeroth_grads[task] = GradEstimator(model, args.zo_eps).forward(
                    x, y, task)

            ############################## SAM, Stage I ##############################
            shared_params = {}
            for n, p in model.shared_base.named_parameters():
                shared_params[n] = p.data.clone()
            task_params = defaultdict(dict)

            shared_epsilon_params = defaultdict(dict)
            task_epsilon_params = defaultdict(dict)
            for task in range(n_tasks):
                task_norms = dict()
                for n, p in model.out_layer[task].named_parameters():
                    task_ep = torch.zeros_like(p).data.clone()
                    if p.grad is not None:
                        task_params[task][n] = p.data.clone()
                        task_norms[n] = (
                            (torch.abs(p) if args.adaptive else 1.0) *
                            p.grad).norm(p=2).data.clone()
                        task_ep = (
                            (torch.pow(p, 2) if args.adaptive else 1.0) *
                            p.grad).data.clone()
                    task_epsilon_params[task][n] = task_ep

                task_norm = torch.norm(torch.stack(list(task_norms.values())),
                                       p=2)
                task_scale = (args.rho / (task_norm + 1e-12)).item()
                task_epsilon_params[task] = {
                    n: ep * task_scale
                    for n, ep in task_epsilon_params[task].items()
                }

                # Get the shared gradient and perturbation
                shared_norms = dict()
                for n, p in model.shared_base.named_parameters():
                    shared_ep = torch.zeros_like(p).data.clone()
                    if p.grad is not None:
                        n1 = p.grad.norm(p=2).item()
                        n2 = zeroth_grads[task][n].norm(p=2).item()
                        # Here (1 - beta) corresponds alpha in our paper
                        g = (1 - args.beta) * p.grad.data.clone(
                        ) + args.beta * n1 * (zeroth_grads[task][n] / n2)
                        shared_norms[n] = (
                            (torch.abs(p) if args.adaptive else 1.0) *
                            g).norm(p=2).data.clone()
                        shared_ep = (
                            (torch.pow(p, 2) if args.adaptive else 1.0) *
                            g).data.clone()
                    shared_epsilon_params[task][n] = shared_ep

                shared_norm = torch.norm(torch.stack(
                    list(shared_norms.values())),
                                         p=2)
                shared_scale = (args.rho / (shared_norm + 1e-12)).item()
                shared_epsilon_params[task] = {
                    n: ep * shared_scale
                    for n, ep in shared_epsilon_params[task].items()
                }

            del task_norms, shared_norms

            ############################## SAM, Stage II ##############################
            disable_running_stats(model)
            shared_sam_grad = defaultdict(dict)
            model.zero_grad()
            for task in range(n_tasks):
                for n, p in model.out_layer[task].named_parameters():
                    p.data = (task_params[task][n] +
                              task_epsilon_params[task][n]).data.clone()
                for n, p in model.shared_base.named_parameters():
                    p.data = (shared_params[n] +
                              shared_epsilon_params[task][n]).data.clone()

                y_ = model(x)
                loss = torch.stack([
                    loss_fn(y_task_pred, y_task)
                    for (y_task_pred, y_task) in zip(y_, y)
                ])[task]
                loss.backward()

                # Restore the task parameters
                # Here only restore task parameters, as we don't need restore the shared parameters multiple times
                for n, p in model.out_layer[task].named_parameters():
                    p.data = task_params[task][n].data.clone()
                for n, p in model.shared_base.named_parameters():
                    if p.grad is not None:
                        shared_sam_grad[task][n] = p.grad.data.clone()
                        p.grad.zero_()

            del task_epsilon_params, shared_epsilon_params

            # Restore the shared parameters
            for n, p in model.shared_base.named_parameters():
                p.data = shared_params[n]

            loss, extra_outputs = weight_method.backward(
                losses=None,
                shared_grads=shared_sam_grad,
                shared_parameters=list(model.shared_parameters()),
                task_specific_parameters=list(
                    model.task_specific_parameters()),
                last_shared_parameters=list(model.last_shared_parameters()),
            )
            optimizer.step()
            if "famo" in args.method:
                with torch.no_grad():
                    y_ = model(x)
                    new_losses = torch.stack([
                        loss_fn(y_task_pred, y_task)
                        for (y_task_pred, y_task) in zip(y_, y)
                    ])
                    weight_method.method.update(new_losses.detach())
        t1 = time.time()

        model.eval()
        # validation
        metric.reset()
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = [y_.to(device) for y_ in y]
                y_ = model(x)
                losses = torch.stack([
                    loss_fn(y_task_pred, y_task)
                    for (y_task_pred, y_task) in zip(y_, y)
                ])
                metric.incr(y_, y)
        val_f1 = metric.result()
        if val_f1.mean() > best_val_f1:
            best_val_f1 = val_f1.mean()
            best_epoch = epoch

        # testing
        metric.reset()
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = [y_.to(device) for y_ in y]
                y_ = model(x)
                losses = torch.stack([
                    loss_fn(y_task_pred, y_task)
                    for (y_task_pred, y_task) in zip(y_, y)
                ])
                metric.incr(y_, y)
        test_f1 = metric.result()
        metrics[epoch] = test_f1

        t2 = time.time()
        print(
            f"[info] epoch {epoch+1} | train takes {(t1-t0)/60:.1f} min | test takes {(t2-t1)/60:.1f} min"
        )
        name = f"{args.method}"

        torch.save({
            "metric": metrics,
            "best_epoch": best_epoch
        }, f"./save/{name}_rho{args.rho}_beta{args.beta}_sd{args.seed}.stats")


if __name__ == "__main__":
    parser = ArgumentParser("Celeba", parents=[common_parser])
    parser.set_defaults(
        data_path=os.path.join(os.getcwd(), "dataset"),
        lr=3e-4,
        n_epochs=15,
        batch_size=256,
    )
    parser.add_argument("--rho",
                        type=float,
                        default=0.001,
                        help="Rho for pertubation in SAM.")
    parser.add_argument("--adaptive",
                        type=str2bool,
                        default=False,
                        help="Adaptive SAM.")
    parser.add_argument(
        "--beta",
        default=0.1,
        type=float,
        help="Interpolation coefficient for perturbation term.")
    parser.add_argument("--zo_eps",
                        default=0.01,
                        type=float,
                        help="Epsilon for zeroth order gradient estimation.")
    args = parser.parse_args()

    # set seed
    set_seed(args.seed)
    device = get_device(gpus=args.gpu)
    main(path=args.data_path,
         lr=args.lr,
         bs=args.batch_size,
         args=args,
         device=device)
