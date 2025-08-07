from math import e
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import sys
from datetime import datetime
from typing import Dict

import monai
import torch
import yaml
from tqdm import tqdm
import torch.nn as nn
from dataclasses import dataclass, field
from accelerate import Accelerator
from easydict import EasyDict
from monai.utils import ensure_tuple_rep
from objprint import objstr
from timm.optim import optim_factory
from accelerate.utils import DistributedDataParallelKwargs

from src import utils
from src.loader import get_dataloader_GCNC as get_dataloader
from src.optimizer import LinearWarmupCosineAnnealingLR
from src.utils import (
    Logger,
    write_example,
    resume_train_state,
    split_metrics,
    load_model_dict,
    freeze_seg_decoder,
    reload_pre_train_model,
)
from src.eval import (
    calculate_f1_score,
    specificity,
    quadratic_weighted_kappa,
    top_k_accuracy,
    calculate_metrics,
    accumulate_metrics,
    compute_final_metrics,
)

from get_model import get_model


def train_one_epoch(
    model: torch.nn.Module,
    loss_functions: Dict[str, torch.nn.modules.loss._Loss],
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
    post_trans: monai.transforms.Compose,
    accelerator: Accelerator,
    epoch: int,
    step: int,
    config: EasyDict,
):
    # 训练
    model.train()
    freeze_seg_decoder(model)
    accelerator.print(f"Training...", flush=True)
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    # for i, image_batch in enumerate(train_loader):
    for i, image_batch in loop:
        # for i, image_batch in enumerate(train_loader):
        if config.trainer.choose_model == "HSL_Net":
            logits, _ = model(image_batch["image"])
        else:
            logits = model(image_batch["image"])
        total_loss = 0
        logits_loss = logits
        labels = image_batch["m_label"]
        for name in loss_functions:
            alpth = 1
            loss = loss_functions[name](logits_loss, labels.float())
            accelerator.log({"Train/" + name: float(loss)}, step=step)
            total_loss += alpth * loss

        for metric_name in metrics:
            y_pred = post_trans(logits)
            y = labels
            if metric_name == "miou_metric":
                y_pred = y_pred.unsqueeze(2)
                y = y.unsqueeze(2)
            metrics[metric_name](y_pred=y_pred, y=y)

        accelerator.backward(total_loss)
        optimizer.step()
        optimizer.zero_grad()
        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(name)
        accelerator.log(
            {
                "Train/Total Loss": float(total_loss),
            },
            step=step,
        )
        # accelerator.print(
        #     f'Epoch [{epoch+1}/{config.trainer.num_epochs}][{i + 1}/{len(train_loader)}] Training Loss:{total_loss}',
        #     flush=True
        #     )
        # 更新信息
        loop.set_description(f"Epoch [{epoch+1}/{config.trainer.num_epochs}]")
        loop.set_postfix(loss=total_loss)
        step += 1
    scheduler.step(epoch)

    metric = {}
    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate()[0].to(accelerator.device)

        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes

        # give every single task metric
        metrics[metric_name].reset()
        metric.update(
            {
                f"Train/{metric_name}": float(batch_acc.mean()),
            }
        )
    for metric_name in metrics:
        all_data = []
        for key in metric.keys():
            if metric_name in key:
                all_data.append(metric[key])
        me_data = sum(all_data) / len(all_data)
        metric.update({f"Train/{metric_name}": float(me_data)})

    accelerator.log(metric, step=epoch)
    return metric, step


@torch.no_grad()
def val_one_epoch(
    model: torch.nn.Module,
    inference: monai.inferers.Inferer,
    val_loader: torch.utils.data.DataLoader,
    metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
    step: int,
    post_trans: monai.transforms.Compose,
    accelerator: Accelerator,
    config: EasyDict,
    test: bool = False,
):
    # 验证
    model.eval()
    if test:
        flag = "Test"
        accelerator.print(f"Testing...", flush=True)
    else:
        flag = "Val"
        accelerator.print(f"Valing...", flush=True)
    loop = tqdm(enumerate(val_loader), total=len(val_loader))
    # for i, image_batch in enumerate(val_loader):
    for i, image_batch in loop:
        # logits = inference(model, image_batch['image'])
        if config.trainer.choose_model == "HSL_Net":
            logits, _ = model(image_batch["image"])
        else:
            logits = model(image_batch["image"])
        log = ""
        total_loss = 0

        logits_loss = logits
        labels_loss = image_batch["m_label"]

        for name in loss_functions:
            loss = loss_functions[name](logits_loss, labels_loss.float())
            accelerator.log({f"{flag}/" + name: float(loss)}, step=step)
            log += f"{name} {float(loss):1.5f} "
            total_loss += loss

        accelerator.log(
            {
                f"{flag}/Total Loss": float(total_loss),
            },
            step=step,
        )

        for metric_name in metrics:
            y_pred = post_trans(logits)
            y = labels_loss
            if metric_name == "miou_metric":
                y_pred = y_pred.unsqueeze(2)
                y = y.unsqueeze(2)
            metrics[metric_name](y_pred=y_pred, y=y)

        # accelerator.print(
        #     f'[{i + 1}/{len(val_loader)}] {flag} Validation Loading...',
        #     flush=True)
        loop.set_description(f"Epoch [{epoch+1}/{config.trainer.num_epochs}]")
        loop.set_postfix(loss=total_loss)
        step += 1
    metric = {}

    for metric_name in metrics:
        # for channel in range(channels):
        batch_acc = metrics[metric_name].aggregate()[0].to(accelerator.device)

        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes

        # give every single task metric
        metrics[metric_name].reset()
        # task_num = channel + 1
        metric.update(
            {
                f"{flag}/{metric_name}": float(batch_acc.mean()),
            }
        )

    accelerator.log(metric, step=epoch)
    return metric, step


if __name__ == "__main__":
    config = EasyDict(
        yaml.load(open("config.yml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    )
    utils.same_seeds(50)
    if config.finetune.GCNC.checkpoint != "None":
        checkpoint_name = config.finetune.GCNC.checkpoint
    else:
        checkpoint_name = (
            config.trainer.choose_dataset
            + "_"
            + config.trainer.task
            + config.trainer.choose_model
        )

    logging_dir = (
        os.getcwd()
        + "/logs/"
        + checkpoint_name
        + str(datetime.now())
        .replace(" ", "_")
        .replace("-", "_")
        .replace(":", "_")
        .replace(".", "_")
    )
    accelerator = Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
        cpu=False,
        log_with=["tensorboard"],
        project_dir=logging_dir,
    )
    Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.print(objstr(config))

    accelerator.print("load model...")
    model = get_model(config)

    if config.trainer.choose_model == "HSL_Net":
        reload_pre_train_model(model, accelerator, "HSL_Net_class_multimodals_v1")

    accelerator.print("load dataset...")
    train_loader, val_loader, test_loader, example = get_dataloader(config)

    # keep example log
    if accelerator.is_main_process == True:
        write_example(example, logging_dir)

    inference = monai.inferers.SlidingWindowInferer(
        roi_size=config.GCNC_loader.target_size,
        overlap=0.5,
        sw_device=accelerator.device,
        device=accelerator.device,
    )

    loss_functions = {
        "focal_loss": monai.losses.FocalLoss(to_onehot_y=False),
        "bce_loss": nn.BCEWithLogitsLoss().to(accelerator.device),
    }

    metrics = {
        "accuracy": monai.metrics.ConfusionMatrixMetric(
            include_background=False, metric_name="accuracy"
        ),
        "f1": monai.metrics.ConfusionMatrixMetric(
            include_background=False, metric_name="f1 score"
        ),
        "specificity": monai.metrics.ConfusionMatrixMetric(
            include_background=False, metric_name="specificity"
        ),
        "recall": monai.metrics.ConfusionMatrixMetric(
            include_background=False, metric_name="recall"
        ),
        "miou_metric": monai.metrics.MeanIoU(include_background=False),
    }

    post_trans = monai.transforms.Compose(
        [
            monai.transforms.Activations(sigmoid=True),
            monai.transforms.AsDiscrete(threshold=0.5),
        ]
    )

    optimizer = optim_factory.create_optimizer_v2(
        model,
        opt=config.trainer.optimizer,
        weight_decay=float(config.trainer.weight_decay),
        lr=float(config.trainer.lr),
        betas=(config.trainer.betas[0], config.trainer.betas[1]),
    )

    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=config.trainer.warmup,
        max_epochs=config.trainer.num_epochs,
    )

    # start training
    accelerator.print("Start Training! ")
    train_step = 0
    best_eopch = -1
    val_step = 0
    best_accuracy = torch.tensor(0)
    best_metrics = {}
    best_test_accuracy = torch.tensor(0)
    best_test_metrics = {}

    starting_epoch = 0

    if config.trainer.resume:
        (
            model,
            optimizer,
            scheduler,
            starting_epoch,
            train_step,
            best_accuracy,
            best_test_accuracy,
            best_metrics,
            best_test_metrics,
        ) = utils.resume_train_state(
            model,
            "{}".format(checkpoint_name),
            optimizer,
            scheduler,
            train_loader,
            accelerator,
            seg=False,
        )
        val_step = train_step

    model, optimizer, scheduler, train_loader, val_loader, test_loader = (
        accelerator.prepare(
            model, optimizer, scheduler, train_loader, val_loader, test_loader
        )
    )

    best_accuracy = torch.Tensor([best_accuracy]).to(accelerator.device)
    best_test_accuracy = torch.Tensor([best_test_accuracy]).to(accelerator.device)

    for epoch in range(starting_epoch, config.trainer.num_epochs):
        train_metric, train_step = train_one_epoch(
            model,
            loss_functions,
            train_loader,
            optimizer,
            scheduler,
            metrics,
            post_trans,
            accelerator,
            epoch,
            train_step,
            config,
        )

        final_metrics, val_step = val_one_epoch(
            model,
            inference,
            val_loader,
            metrics,
            val_step,
            post_trans,
            accelerator,
            config,
            False,
        )

        val_top = final_metrics["Val/accuracy"]

        # 保存模型
        if val_top > best_accuracy:
            accelerator.save_state(
                output_dir=f"{os.getcwd()}/model_store/{checkpoint_name}/best"
            )
            best_accuracy = final_metrics["Val/accuracy"]
            best_metrics = final_metrics
            # 记录最优test acc
            if config.GCNC_loader.fusion == False:
                final_metrics, _ = val_one_epoch(
                    model,
                    inference,
                    test_loader,
                    metrics,
                    -1,
                    post_trans,
                    accelerator,
                    config,
                    test=True,
                )
                best_test_accuracy = final_metrics["Test/accuracy"]
                best_test_metrics = final_metrics
            else:
                final_metrics = final_metrics

                best_test_accuracy = final_metrics["Val/accuracy"]
                best_test_metrics = final_metrics

        accelerator.print(
            f'Epoch [{epoch+1}/{config.trainer.num_epochs}] now train acc: {train_metric["Train/accuracy"]}, now val acc: {val_top}, best acc: {best_accuracy}, best test acc: {best_test_accuracy}'
        )

        accelerator.print("Cheakpoint...")
        accelerator.save_state(
            output_dir=f"{os.getcwd()}/model_store/{checkpoint_name}/checkpoint"
        )
        torch.save(
            {
                "epoch": epoch,
                "best_accuracy": best_accuracy,
                "best_metrics": best_metrics,
                "best_test_accuracy": best_test_accuracy,
                "best_test_metrics": best_test_metrics,
            },
            f"{os.getcwd()}/model_store/{checkpoint_name}/checkpoint/epoch.pth.tar",
        )

    accelerator.print(f"best test accuracy: {best_test_accuracy}")
    accelerator.print(f"best metrics: {best_test_metrics}")
