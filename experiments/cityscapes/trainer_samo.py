import os
import logging
import wandb
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange

from experiments.cityscapes.data import Cityscapes
from experiments.cityscapes.models import SegNet, SegNetMtan
from experiments.cityscapes.utils import delta_fn, depth_error
from experiments.cityscapes.utils import ConfMatrix, GradEstimator
from experiments.utils import (common_parser,
                               extract_weight_method_parameters_from_args,
                               get_device, set_logger, set_seed, str2bool,
                               enable_running_stats, disable_running_stats)
from methods.weight_methods import WeightMethods
from collections import defaultdict

set_logger()


def calc_loss(x_pred, x_output, task_type):
    device = x_pred.device

    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(x_output, dim=1)
                   != 0).float().unsqueeze(1).to(device)

    if task_type == "semantic":
        # semantic loss: depth-wise cross entropy
        loss = F.nll_loss(x_pred, x_output, ignore_index=-1)

    if task_type == "depth":
        # depth loss: l1 norm
        loss = torch.sum(
            torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(
                binary_mask, as_tuple=False).size(0)

    return loss


def main(args, device):
    # ----
    # Nets
    # ---
    model = dict(segnet=SegNet(), mtan=SegNetMtan())[args.model]
    model = model.to(device)

    # dataset and dataloaders
    log_str = ("Applying data augmentation on Cityscapes."
               if args.apply_augmentation else
               "Standard training strategy without data augmentation.")
    logging.info(log_str)

    cityscapes_train_set = Cityscapes(root=args.data_path.as_posix(),
                                      train=True,
                                      augmentation=args.apply_augmentation)
    cityscapes_test_set = Cityscapes(root=args.data_path.as_posix(),
                                     train=False)

    train_loader = torch.utils.data.DataLoader(dataset=cityscapes_train_set,
                                               batch_size=args.batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=cityscapes_test_set,
                                              batch_size=args.batch_size,
                                              shuffle=False)

    # weight method
    n_tasks = 2
    weight_methods_parameters = extract_weight_method_parameters_from_args(
        args)
    weight_method = WeightMethods(args.method,
                                  n_tasks=2,
                                  device=device,
                                  **weight_methods_parameters[args.method])

    # optimizer
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=args.lr),
        dict(params=weight_method.parameters(), lr=args.method_params_lr),
    ], )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=100,
                                                gamma=0.5)

    epochs = args.n_epochs
    epoch_iter = trange(epochs)
    train_batch = len(train_loader)
    test_batch = len(test_loader)
    avg_cost = np.zeros([epochs, 12], dtype=np.float32)
    custom_step = -1
    conf_mat = ConfMatrix(model.segnet.class_nb)
    deltas = np.zeros([
        epochs,
    ], dtype=np.float32)

    # some extra statistics we save during training
    loss_list = []

    for epoch in epoch_iter:
        cost = np.zeros(12, dtype=np.float32)

        for j, batch in enumerate(train_loader):
            custom_step += 1

            model.train()

            train_data, train_label, train_depth = batch
            train_data, train_label = train_data.to(
                device), train_label.long().to(device)
            train_depth = train_depth.to(device)

            enable_running_stats(model)
            train_pred, features = model(train_data,
                                         return_representation=True)

            losses = torch.stack((
                calc_loss(train_pred[0], train_label, "semantic"),
                calc_loss(train_pred[1], train_depth, "depth"),
            ))
            # Get the average gradient
            losses.mean().backward()

            # Get the estimated gradient
            zeroth_grads = {}
            for task in range(n_tasks):
                if task == 0:
                    targets = train_label
                elif task == 1:
                    targets = train_depth
                zeroth_grads[task] = GradEstimator(
                    model, eps=args.zo_eps).forward(train_data, targets, task)

            ############################## SAM, Stage I ##############################
            shared_params = dict()
            for n, p in model.named_parameters():
                if "pred" not in n:
                    shared_params[n] = p.data.clone()
            task_params = defaultdict(dict)

            # global normalization
            # global_grad_norms = []
            # for n, p in model.named_parameters():
            #     if "pred" in n:
            #         continue
            #     global_grad_norms.append(p.grad.norm(p=2).data.clone())
            # global_grad_norm = torch.norm(torch.stack(global_grad_norms), p=2)

            # task_grad_norms = defaultdict(list)
            # task_grad_norm = {}
            # for task in range(n_tasks):
            #     for n, pg in zeroth_grads[task].items():
            #         task_grad_norms[task].append(pg.norm(p=2).data.clone())
            #     task_grad_norm[task] = torch.norm(torch.stack(task_grad_norms[task]), p=2)

            shared_epsilon_params = defaultdict(dict)
            task_epsilon_params = defaultdict(dict)
            for task in range(n_tasks):

                # Get the task specific gradient and perturbation
                task_norms = dict()
                for n, p in model.named_parameters():
                    if f"pred_task{task + 1}" in n:
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
                    if "pred" in n:
                        continue
                    shared_ep = torch.zeros_like(p).data
                    if p.grad is not None:
                        n1 = p.grad.norm(p=2).item()
                        n2 = zeroth_grads[task][n].norm(p=2).item()
                        g = (1 - args.beta) * p.grad.data.clone(
                        ) + args.beta * n1 * (zeroth_grads[task][n] / n2)
                        # g = (1 - args.beta) * p.grad.data.clone() + args.beta * global_grad_norm * (zeroth_grads[task][n] / task_grad_norm[task])
                        # g = (1 - args.beta) * p.grad.data.clone() + args.beta * zeroth_grads[task][n]
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
                for n, p in model.named_parameters():
                    if "pred" in n:
                        if f"pred_task{task + 1}" in n:
                            p.data = (
                                task_params[task][n] +
                                task_epsilon_params[task][n]).data.clone()
                    else:
                        if p.grad is not None:
                            p.grad.zero_()
                        p.data = (shared_params[n] +
                                  shared_epsilon_params[task][n]).data.clone()

                train_pred, _ = model(train_data, return_representation=True)

                if task == 0:
                    calc_loss(train_pred[0], train_label,
                              "semantic").backward()
                elif task == 1:
                    calc_loss(train_pred[1], train_depth, "depth").backward()
                else:
                    raise ValueError(f"Task {task} not supported")

                # Restore the task parameters
                # Here only restore task parameters, as we don't need restore shared parameters multiple times
                for n, p in model.named_parameters():
                    if "pred" in n:
                        if f"pred_task{task + 1}" in n:
                            p.data = task_params[task][n].data.clone()
                        continue
                    # Get the shared gradient of each task
                    if p.grad is not None:
                        shared_sam_grad[task][n] = p.grad.data.clone()
                        p.grad.zero_()

            del task_epsilon_params, shared_epsilon_params

            # Restore the shared parameters
            for n, p in model.named_parameters():
                if "pred" not in n:
                    p.data = shared_params[n]

            loss, extra_outputs = weight_method.backward(
                losses=None,
                shared_grads=shared_sam_grad,
                shared_parameters=list(model.shared_parameters()),
                task_specific_parameters=list(
                    model.task_specific_parameters()),
                last_shared_parameters=list(model.last_shared_parameters()),
                representation=features,
            )

            loss_list.append(losses.detach().cpu())
            optimizer.step()

            if "famo" in args.method:
                with torch.no_grad():
                    train_pred = model(train_data, return_representation=False)
                    new_losses = torch.stack((
                        calc_loss(train_pred[0], train_label, "semantic"),
                        calc_loss(train_pred[1], train_depth, "depth"),
                    ))
                    weight_method.method.update(new_losses.detach())

            # accumulate label prediction for every pixel in training images
            conf_mat.update(train_pred[0].argmax(1).flatten(),
                            train_label.flatten())

            cost[0] = losses[0].item()
            cost[3] = losses[1].item()
            cost[4], cost[5] = depth_error(train_pred[1], train_depth)
            avg_cost[epoch, :6] += cost[:6] / train_batch

            epoch_iter.set_description(
                f"[{epoch+1}  {j+1}/{train_batch}] semantic loss: {losses[0].item():.3f}, "
                f"depth loss: {losses[1].item():.3f}, ")

        # scheduler
        scheduler.step()
        # compute mIoU and acc
        avg_cost[epoch, 1:3] = conf_mat.get_metrics()

        # todo: move evaluate to function?
        # evaluating test data
        model.eval()
        conf_mat = ConfMatrix(model.segnet.class_nb)
        with torch.no_grad():  # operations inside don't track history
            test_dataset = iter(test_loader)
            for k in range(test_batch):
                test_data, test_label, test_depth = test_dataset.next()
                test_data, test_label = test_data.to(
                    device), test_label.long().to(device)
                test_depth = test_depth.to(device)

                test_pred = model(test_data)
                test_loss = torch.stack((
                    calc_loss(test_pred[0], test_label, "semantic"),
                    calc_loss(test_pred[1], test_depth, "depth"),
                ))

                conf_mat.update(test_pred[0].argmax(1).flatten(),
                                test_label.flatten())

                cost[6] = test_loss[0].item()
                cost[9] = test_loss[1].item()
                cost[10], cost[11] = depth_error(test_pred[1], test_depth)
                avg_cost[epoch, 6:] += cost[6:] / test_batch

            # compute mIoU and acc
            avg_cost[epoch, 7:9] = conf_mat.get_metrics()

            # Test Delta_m
            test_delta_m = delta_fn(avg_cost[epoch, [7, 8, 10, 11]])
            deltas[epoch] = test_delta_m

            # print results
            print(
                f"LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR "
            )
            print(
                f"Epoch: {epoch:04d} | TRAIN: {avg_cost[epoch, 0]:.4f} {avg_cost[epoch, 1]:.4f} {avg_cost[epoch, 2]:.4f} "
                f"| {avg_cost[epoch, 3]:.4f} {avg_cost[epoch, 4]:.4f} {avg_cost[epoch, 5]:.4f} | {avg_cost[epoch, 6]:.4f} "
                f"TEST: {avg_cost[epoch, 7]:.4f} {avg_cost[epoch, 8]:.4f} {avg_cost[epoch, 9]:.4f} | "
                f"{avg_cost[epoch, 10]:.4f} {avg_cost[epoch, 11]:.4f}"
                f"| {test_delta_m:.3f}")

            if wandb.run is not None:
                wandb.log({"Train Semantic Loss": avg_cost[epoch, 0]},
                          step=epoch)
                wandb.log({"Train Mean IoU": avg_cost[epoch, 1]}, step=epoch)
                wandb.log({"Train Pixel Accuracy": avg_cost[epoch, 2]},
                          step=epoch)
                wandb.log({"Train Depth Loss": avg_cost[epoch, 3]}, step=epoch)
                wandb.log({"Train Absolute Error": avg_cost[epoch, 4]},
                          step=epoch)
                wandb.log({"Train Relative Error": avg_cost[epoch, 5]},
                          step=epoch)

                wandb.log({"Test Semantic Loss": avg_cost[epoch, 6]},
                          step=epoch)
                wandb.log({"Test Mean IoU": avg_cost[epoch, 7]}, step=epoch)
                wandb.log({"Test Pixel Accuracy": avg_cost[epoch, 8]},
                          step=epoch)
                wandb.log({"Test Depth Loss": avg_cost[epoch, 9]}, step=epoch)
                wandb.log({"Test Absolute Error": avg_cost[epoch, 10]},
                          step=epoch)
                wandb.log({"Test Relative Error": avg_cost[epoch, 11]},
                          step=epoch)
                wandb.log({"Test ∆m": test_delta_m}, step=epoch)

            keys = [
                "Train Semantic Loss",
                "Train Mean IoU",
                "Train Pixel Accuracy",
                "Train Depth Loss",
                "Train Absolute Error",
                "Train Relative Error",
                "Test Semantic Loss",
                "Test Mean IoU",
                "Test Pixel Accuracy",
                "Test Depth Loss",
                "Test Absolute Error",
                "Test Relative Error",
            ]
            name = f"{args.method}_rho{args.rho}_sd{args.seed}"

            torch.save(
                {
                    "delta_m": deltas,
                    "keys": keys,
                    "avg_cost": avg_cost,
                    "losses": loss_list,
                }, f"./save/{name}.stats")


if __name__ == "__main__":
    parser = ArgumentParser("Cityscapes", parents=[common_parser])
    parser.set_defaults(
        data_path=os.path.join(os.getcwd(), "dataset"),
        lr=1e-4,
        n_epochs=200,
        batch_size=8,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mtan",
        choices=["segnet", "mtan"],
        help="model type",
    )
    parser.add_argument("--apply-augmentation",
                        type=str2bool,
                        default=True,
                        help="data augmentations")
    # sharpness
    parser.add_argument("--rho",
                        type=float,
                        default=0.003,
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
    parser.add_argument("--wandb_project",
                        type=str,
                        default=None,
                        help="Name of Weights & Biases Project.")
    parser.add_argument("--wandb_entity",
                        type=str,
                        default=None,
                        help="Name of Weights & Biases Entity.")
    args = parser.parse_args()

    # set seed
    set_seed(args.seed)

    if args.wandb_project is not None:
        wandb.init(project=args.wandb_project,
                   entity=args.wandb_entity,
                   config=args)

    device = get_device(gpus=args.gpu)
    main(args=args, device=device)

    if wandb.run is not None:
        wandb.finish()
