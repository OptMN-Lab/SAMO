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
from experiments.office_home.models import SingleTaskNet
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
                eval_pred = model(eval_input)
                eval_losses[tn] = loss_fns[task].update_loss(
                    eval_pred, eval_gt)
                metrics[task].update_fun(eval_pred, eval_gt)
    eval_dict = defaultdict(dict)
    for tn, task in enumerate(task_name):
        eval_dict['score'][task] = metrics[task].score_fun()
        eval_dict['loss'][task] = loss_fns[task].average_loss()
    return eval_dict


def main(args, device):
    # task_name = ['Art', 'Clipart', 'Product', 'Real_World']
    task_name = ['Art']
    n_tasks = len(task_name)
    model = SingleTaskNet()
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

    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=args.lr,
                                 weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=50,
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

            train_input, train_gt = process_data(train_loader[task], device)
            train_pred = model(train_input)
            train_loss = loss_fns[task].update_loss(train_pred, train_gt)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            epoch_iter.set_description(
                f"[{epoch+1}  {batch_index+1}/{train_batch}] " +
                ", ".join([f"{task_name[0]} loss: {train_loss.item():.3f}"]))

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
        print(f"\nEpoch: {epoch + 1:04d} | "
              f"VAL LOSS: " +
              " ".join([f"{val_dict['loss'][task]:.3f}"
                        for task in task_name]) + " | "
              f"TEST LOSS: " + " ".join([
                  f"{task} {test_dict['loss'][task]:.3f}" for task in task_name
              ]) + " | "
              f"BEST TEST AVG SCORE: {best_test_score:.3f}\n")

        name = f"{task_name[0]}_sd{args.seed}"
        torch.save(
            {
                "val_score": best_val_score,
                "test_score": best_test_score,
                "val_dict": best_val_dict,
                "test_dict": best_test_dict,
            }, f"./single/{name}.stats")


if __name__ == "__main__":
    parser = ArgumentParser("OfficeHome", parents=[common_parser])
    parser.set_defaults(
        data_path=os.path.join(os.getcwd(), "dataset"),
        lr=1e-3,
        n_epochs=100,
        batch_size=64,
    )
    args = parser.parse_args()

    # set seed
    set_seed(args.seed)

    device = get_device(gpus=args.gpu)
    main(args=args, device=device)
