from argparse import ArgumentParser
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import trange
from collections import defaultdict

from experiments.office_home.data import office_dataloader
from experiments.office_home.models import Net
from experiments.utils import (common_parser,
                               extract_weight_method_parameters_from_args,
                               get_device, set_logger, set_seed, str2bool,
                               enable_running_stats, disable_running_stats)
from experiments.office_home.utils import AccMetric, CELoss, GradEstimator
from experiments.office_home.utils import prepare_dataloaders, process_data
from methods.weight_methods import WeightMethods

set_logger()


@torch.no_grad()
def evaluate(model, device, dataloaders, task_name, loss_fns, metrics):
    model.eval()
    eval_loader, eval_batch = prepare_dataloaders(dataloaders, task_name)
    eval_batch = max(eval_batch)
    with torch.no_grad():
        for batch_index in range(eval_batch):
            eval_losses = torch.zeros(len(task_name)).to(device)
            for tn, task in enumerate(task_name):
                eval_input, eval_gt = process_data(eval_loader[task], device)
                eval_pred = model(eval_input, task)
                eval_losses[tn] = loss_fns[task].update_loss(
                    eval_pred, eval_gt)
                metrics[task].update_fun(eval_pred, eval_gt)
    eval_dict = defaultdict(dict)
    for tn, task in enumerate(task_name):
        eval_dict['score'][task] = metrics[task].score_fun()
        eval_dict['loss'][task] = loss_fns[task].average_loss()
    return eval_dict


def main(args, device):
    task_name = ['Art', 'Clipart', 'Product', 'Real_World']
    n_tasks = len(task_name)
    model = Net()
    model = model.to(device)

    # prepare dataloaders
    log_str = "Loading office home dataset."
    logging.info(log_str)

    data_loader, _ = office_dataloader(batchsize=args.batch_size,
                                       root_path=args.data_path)
    train_dataloaders = {
        task: data_loader[task]['train']
        for task in task_name
    }
    val_dataloaders = {task: data_loader[task]['val'] for task in task_name}
    test_dataloaders = {task: data_loader[task]['test'] for task in task_name}

    loss_fns = {task: CELoss() for task in task_name}
    metrics = {task: AccMetric() for task in task_name}
    sam_loss_fn = nn.CrossEntropyLoss()

    # weight method
    weight_methods_parameters = extract_weight_method_parameters_from_args(
        args)

    weight_method = WeightMethods(args.method,
                                  n_tasks=n_tasks,
                                  device=device,
                                  **weight_methods_parameters[args.method])

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=args.lr, weight_decay=1e-5),
        dict(params=weight_method.parameters(), lr=args.method_params_lr),
    ], )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=25,
                                                gamma=0.5)

    train_loader, train_batch = prepare_dataloaders(train_dataloaders,
                                                    task_name)
    train_batch = max(train_batch)

    best_val_score = 0
    best_test_score = 0
    best_val_dict = None
    best_test_dict = None
    epoch_iter = trange(args.n_epochs)
    for epoch in epoch_iter:
        model.train()
        for task in task_name:
            metrics[task].reinit()
            loss_fns[task].reinit()
        for batch_index in range(train_batch):

            train_losses = torch.zeros(n_tasks).to(device)
            sam_data = {}
            enable_running_stats(model)
            for tn, task in enumerate(task_name):
                train_input, train_gt = process_data(train_loader[task],
                                                     device)
                sam_data[task] = (train_input, train_gt)
                train_pred = model(train_input, task)
                train_losses[tn] = loss_fns[task].update_loss(
                    train_pred, train_gt)
                # metrics[task].update_fun(train_pred, train_gt)

            optimizer.zero_grad()

            # Get the average gradient
            train_losses.mean().backward()

            # Get the estimated gradient
            zeroth_grads = {}
            for task in task_name:
                zeroth_grads[task] = GradEstimator(model,
                                                   eps=args.zo_eps).forward(
                                                       sam_data[task], task)

            ############################## SAM, Stage I ##############################
            shared_params = dict()
            for n, p in model.named_parameters():
                if "out_layers" not in n:
                    shared_params[n] = p.data.clone()
            task_params = defaultdict(dict)

            shared_epsilon_params = defaultdict(dict)
            task_epsilon_params = defaultdict(dict)
            for task in task_name:
                # Get the task specific gradient and perturbation
                task_norms = dict()
                for n, p in model.named_parameters():
                    if n.startswith(f"out_layers.{task}"):
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
                for n, p in model.named_parameters():
                    if n.startswith("out_layers"):
                        continue
                    shared_ep = torch.zeros_like(p).data
                    if p.grad is not None:
                        n1 = p.grad.norm(p=2).item()
                        n2 = zeroth_grads[task][n].norm(p=2).item()
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
            for task in task_name:
                for n, p in model.named_parameters():
                    if n.startswith("out_layers"):
                        if n.startswith(f"out_layers.{task}"):
                            p.data = (
                                task_params[task][n] +
                                task_epsilon_params[task][n]).data.clone()
                    else:
                        if p.grad is not None:
                            p.grad.zero_()
                        p.data = (shared_params[n] +
                                  shared_epsilon_params[task][n]).data.clone()

                sam_input, sam_gt = sam_data[task]
                sam_pred = model(sam_input, task)
                loss = sam_loss_fn(sam_pred, sam_gt)
                loss.backward()

                # Restore the task parameters
                # Here only restore task parameters, as we don't need restore shared parameters multiple times
                for n, p in model.named_parameters():
                    if n.startswith("out_layers"):
                        if n.startswith(f"out_layers.{task}"):
                            p.data = task_params[task][n].data.clone()
                        continue
                    # Get the shared gradient of each task
                    if p.grad is not None:
                        shared_sam_grad[task][n] = p.grad.data.clone()
                        p.grad.zero_()

            del task_epsilon_params, shared_epsilon_params

            loss, extra_outputs = weight_method.backward(
                losses=None,
                shared_grads=shared_sam_grad,
                shared_parameters=list(model.shared_parameters()),
                task_specific_parameters=list(
                    model.task_specific_parameters()),
                last_shared_parameters=list(model.last_shared_parameters()),
            )
            optimizer.step()

            epoch_iter.set_description(
                f"[{epoch+1}  {batch_index+1}/{train_batch}] " + ", ".join([
                    f"{task} Loss: {train_losses[tn].item():.3f}"
                    for tn, task in enumerate(task_name)
                ]))

        # scheduler
        scheduler.step()

        # evaluating test data
        for task in task_name:
            metrics[task].reinit()
            loss_fns[task].reinit()
        val_dict = evaluate(model, device, val_dataloaders, task_name,
                            loss_fns, metrics)
        for task in task_name:
            metrics[task].reinit()
            loss_fns[task].reinit()
        test_dict = evaluate(model, device, test_dataloaders, task_name,
                             loss_fns, metrics)

        val_score = np.mean(list(val_dict['score'].values()))
        val_loss = np.mean(list(val_dict['loss'].values()))
        test_score = np.mean(list(test_dict['score'].values()))
        test_loss = np.mean(list(test_dict['loss'].values()))
        best_val_criteria = val_score >= best_val_score
        if best_val_criteria:
            best_val_score = val_score
            best_test_score = test_score
            best_val_dict = val_dict
            best_test_dict = test_dict

        # print results
        # print(f"LOSS FORMAT: ART_LOSS CLIPART_LOSS PRODUCT_LOSS REAL_WORLD_LOSS")
        print(
            f"\nEpoch: {epoch + 1:04d} | "
            f"VAL LOSS: " +
            " ".join([f"{val_dict['loss'][task]:.3f}"
                      for task in task_name]) + " | "
            f"TEST LOSS: " +
            " ".join([f"{test_dict['loss'][task]:.3f}"
                      for task in task_name]) + " | "
            f"BEST TEST AVG SCORE: {best_test_score:.3f}\n")

        name = f"{args.method}_rho{args.rho}_beta{args.beta}_sd{args.seed}_v3"
        torch.save(
            {
                "val_score": best_val_score,
                "test_score": best_test_score,
                "val_dict": best_val_dict,
                "test_dict": best_test_dict,
            }, f"./save/{name}.stats")


if __name__ == "__main__":
    parser = ArgumentParser("OfficeHome", parents=[common_parser])
    parser.set_defaults(
        data_path=os.path.join(os.getcwd(), "dataset"),
        lr=1e-4,
        n_epochs=50,
        batch_size=64,
        method="fairgrad",
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
    main(args=args, device=device)
