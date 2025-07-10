from argparse import ArgumentParser
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import trange
from collections import defaultdict
import ipdb

from experiments.office_home.data import office_dataloader
from experiments.office_home.models import Net, CLIPNet
from experiments.utils import (common_parser,
                               extract_weight_method_parameters_from_args,
                               get_device, set_logger, set_seed, str2bool,
                               enable_running_stats, disable_running_stats)
from experiments.office_home.utils import AccMetric, CELoss
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
    # model = Net()
    model = CLIPNet()
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

    cosine_sim_method1 = defaultdict()
    cosine_sim_method2 = defaultdict()

    best_val_score = 0
    best_test_score = 0
    best_val_dict = None
    best_test_dict = None
    epoch_iter = trange(args.n_epochs)
    for epoch in epoch_iter:
        cosine_sim_method1[epoch] = defaultdict(list)
        cosine_sim_method2[epoch] = defaultdict(list)
        model.train()
        for task in task_name:
            metrics[task].reinit()
            loss_fns[task].reinit()
        for batch_index in range(train_batch):

            train_losses = torch.zeros(n_tasks).to(device)
            for tn, task in enumerate(task_name):
                train_input, train_gt = process_data(train_loader[task],
                                                     device)
                train_pred = model(train_input, task)
                train_losses[tn] = loss_fns[task].update_loss(
                    train_pred, train_gt)
                # metrics[task].update_fun(train_pred, train_gt)

            # compute the gradient cosine sim
            grads = defaultdict(dict)
            for tn, task in enumerate(task_name):
                per_grads = list(
                    torch.autograd.grad(train_losses[tn],
                                        model.model.parameters(),
                                        retain_graph=True,
                                        allow_unused=True))
                all_grads = [
                    grad.view(-1) for grad in per_grads if grad is not None
                ]
                all_grads = torch.cat(all_grads)
                per_grads = [grad.clone() for grad in all_grads]
                grads[task]['all_grads'] = all_grads
                # grads[task]['per_grads'] = per_grads

            for i, task_i in enumerate(task_name):
                for j, task_j in enumerate(task_name):
                    if task_i == task_j:
                        continue
                    k = f'{task_i}-{task_j}'
                    sim1 = cosine_similarity_method1(
                        grads[task_i]['all_grads'], grads[task_j]['all_grads'])
                    # sim2 = cosine_similarity_method2(grads[task_i]['per_grads'], grads[task_j]['per_grads'])

                    cosine_sim_method1[epoch][k].append(sim1)
                    # cosine_sim_method2[epoch][k].append(sim2)

            optimizer.zero_grad()
            loss, extra_outputs = weight_method.backward(
                losses=train_losses,
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

        name = f"{args.method}_clip_sd{args.seed}"
        torch.save(
            {
                "val_score": best_val_score,
                "test_score": best_test_score,
                "val_dict": best_val_dict,
                "test_dict": best_test_dict,
                "cosine_sim_method1": cosine_sim_method1,
                # "cosine_sim_method2": cosine_sim_method2,
            },
            f"./save/{name}.stats")


def cosine_similarity_method1(grad1, grad2):
    return torch.nn.functional.cosine_similarity(grad1.unsqueeze(0),
                                                 grad2.unsqueeze(0)).item()


def cosine_similarity_method2(per_param_grads1, per_param_grads2):
    similarities = []
    for g1, g2 in zip(per_param_grads1, per_param_grads2):
        if g1.numel() > 0 and g2.numel() > 0:
            sim = torch.nn.functional.cosine_similarity(
                g1.unsqueeze(0), g2.unsqueeze(0)).item()
    return np.mean(similarities) if similarities else 0.0


if __name__ == "__main__":
    parser = ArgumentParser("OfficeHome", parents=[common_parser])
    parser.set_defaults(
        data_path=os.path.join(os.getcwd(), "dataset"),
        lr=1e-5,
        n_epochs=2,
        batch_size=64,
        method="fairgrad",
    )
    args = parser.parse_args()

    # set seed
    set_seed(args.seed)

    device = get_device(gpus=args.gpu)
    main(args=args, device=device)
