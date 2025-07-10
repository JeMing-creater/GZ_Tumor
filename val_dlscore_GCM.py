import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
import csv
from datetime import datetime
from typing import Dict

import monai
import torch
import yaml
import pandas as pd
import numpy as np
import torch.nn as nn
from dataclasses import dataclass, field
from accelerate import Accelerator
from easydict import EasyDict
from monai.utils import ensure_tuple_rep
from objprint import objstr
from timm.optim import optim_factory

from src import utils
from src.loader import get_dataloader_GCM as get_dataloader
from src.loader import get_GCM_transforms as get_transforms
from src.loader import read_csv_for_GCM
from src.optimizer import LinearWarmupCosineAnnealingLR
from src.utils import (
    Logger,
    write_example,
    resume_train_state,
    split_metrics,
    load_model_dict,
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


def load_model(model, accelerator, checkpoint):
    try:
        check_path = f"{os.getcwd()}/model_store/{checkpoint}/best/"
        accelerator.print("load model from %s" % check_path)
        checkpoint = load_model_dict(
            check_path + "pytorch_model.bin",
        )
        model.load_state_dict(checkpoint)
        accelerator.print(f"Load checkpoint model successfully!")
    except Exception as e:
        accelerator.print(e)
        accelerator.print(f"Failed to load checkpoint model!")
    return model


@torch.no_grad()
def val_one_epoch(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
    post_trans: monai.transforms.Compose,
    accelerator: Accelerator,
):
    # 验证
    model.eval()
    for i, image_batch in enumerate(val_loader):
        logits = model(image_batch["image"])
        log = ""
        total_loss = 0

        logits_loss = logits
        labels_loss = image_batch["class_label"]

        for name in loss_functions:
            loss = loss_functions[name](logits_loss, labels_loss.float())
            log += f"{name}: {float(loss):1.5f} ; "
            total_loss += loss

        log += f"Total Loss: {float(total_loss):1.5f}"

        for metric_name in metrics:
            y_pred = post_trans(logits)
            y = labels_loss
            if metric_name == "miou_metric":
                y_pred = y_pred.unsqueeze(2)
                y = y.unsqueeze(2)
            metrics[metric_name](y_pred=y_pred, y=y)

        accelerator.print(f"[{i + 1}/{len(val_loader)}] {log} ", flush=True)
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
                f"{metric_name}": float(batch_acc.mean()),
            }
        )
    accelerator.print(metric)
    return


@torch.no_grad()
def compute_dl_score_for_example(model, config, post_trans, examples):

    def compute_for_sinlge_example(post_trans, example):
        dl_score = {}
        lable_score = {}
        load_transform, _, _ = get_transforms(config)
        for e in example:
            choose_image = config.GCM_loader.root + "/" + "ALL" + "/" + f"{e}"
            accelerator.print("valing for image: ", choose_image)

            images = []
            labels = []
            for i in range(len(config.GCM_loader.checkModels)):
                image_path = (
                    choose_image
                    + "/"
                    + config.GCM_loader.checkModels[i]
                    + "/"
                    + f"{e}.nii.gz"
                )
                label_path = (
                    choose_image
                    + "/"
                    + config.GCM_loader.checkModels[i]
                    + "/"
                    + f"{e}seg.nii.gz"
                )

                batch = load_transform[i]({"image": image_path, "label": label_path})
                images.append(batch["image"].unsqueeze(1))
                labels.append(batch["label"].unsqueeze(1))

            image_tensor = torch.cat(images, dim=1).to(accelerator.device)
            # label_tensor = torch.cat(labels, dim=1)

            logits = model(image_tensor)
            probs = logits
            dl_s = probs.item()
            l = post_trans(logits).item()
            dl_score[e] = dl_s
            lable_score[e] = l
            # gt_score[e] =
        return dl_score, lable_score

    # 6. 定义一个函数，对 DL-scores 进行正态化
    def normalize_dl_scores(dl_scores):
        """
        正态化 DL-scores，并确保它们保持在 [0, 1] 区间内
        """
        # 计算所有 DL-scores 的均值和标准差
        scores = list(dl_scores.values())  # 获取所有 DL-scores 的值
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        # Z-score 标准化
        standardized_scores = {
            k: (v - mean_score) / std_score for k, v in dl_scores.items()
        }

        # 使用 sigmoid 将值映射回 [0, 1] 范围
        normalized_scores = {
            k: 1 / (1 + np.exp(-v)) for k, v in standardized_scores.items()
        }

        return normalized_scores

    def write_to_csv(dl_score, lable_score, csv_path, use_data_dict):
        # 判断路径是否存在
        dir_path = os.path.dirname(csv_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"dir {dir_path} has been created!")
        else:
            print(f"dir {dir_path} existed!")

        with open(csv_path, mode="w", newline="") as file:
            writer = csv.writer(file)

            # 写入表头
            writer.writerow(["Key", "Value", "Label", "Ground Truth"])

            # 遍历字典并写入键值对
            for key, value in dl_score.items():
                writer.writerow(
                    [str(key), value, lable_score[key], use_data_dict[key][0]]
                )

    def change_to_xlxs(csv_file, save_path):
        df = pd.read_csv(csv_file, dtype={0: str})

        # 清理病人编号列中的前后空格
        df.iloc[:, 0] = df.iloc[:, 0].str.strip()

        # 将数据保存为 Excel 文件
        df.to_excel(save_path, index=False, engine="openpyxl")

        os.remove(csv_file)

    data1, data2, data3 = read_csv_for_GCM(config)

    if config.GCM_loader.task == "PM":
        use_data_dict = data1
    elif config.GCM_loader.task == "NL_SS":
        use_data_dict = data2
    else:
        use_data_dict = data3

    train_example, val_example, test_example = examples
    tr_dl_score, tr_label = compute_for_sinlge_example(post_trans, train_example)
    val_dl_score, val_label = compute_for_sinlge_example(post_trans, val_example)
    te_dl_score, te_label = compute_for_sinlge_example(post_trans, test_example)

    tr_dl_score = normalize_dl_scores(tr_dl_score)
    val_dl_score = normalize_dl_scores(val_dl_score)
    te_dl_score = normalize_dl_scores(te_dl_score)

    write_to_csv(
        tr_dl_score,
        tr_label,
        os.path.join(config.valer.dl_score_csv_path, "train_dl_score.csv"),
        use_data_dict,
    )
    write_to_csv(
        val_dl_score,
        val_label,
        os.path.join(config.valer.dl_score_csv_path, "val_dl_score.csv"),
        use_data_dict,
    )
    write_to_csv(
        te_dl_score,
        te_label,
        os.path.join(config.valer.dl_score_csv_path, "test_dl_score.csv"),
        use_data_dict,
    )

    change_to_xlxs(
        os.path.join(config.valer.dl_score_csv_path, "train_dl_score.csv"),
        os.path.join(config.valer.dl_score_csv_path, "train_dl_score.xlsx"),
    )
    change_to_xlxs(
        os.path.join(config.valer.dl_score_csv_path, "val_dl_score.csv"),
        os.path.join(config.valer.dl_score_csv_path, "val_dl_score.xlsx"),
    )
    change_to_xlxs(
        os.path.join(config.valer.dl_score_csv_path, "test_dl_score.csv"),
        os.path.join(config.valer.dl_score_csv_path, "test_dl_score.xlsx"),
    )

    return tr_dl_score, val_dl_score, te_dl_score


if __name__ == "__main__":
    config = EasyDict(
        yaml.load(open("config.yml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    )
    utils.same_seeds(50)
    logging_dir = (
        os.getcwd()
        + "/logs/"
        + config.finetune.GCM.checkpoint
        + str(datetime.now())
        .replace(" ", "_")
        .replace("-", "_")
        .replace(":", "_")
        .replace(".", "_")
    )
    accelerator = Accelerator(
        cpu=False, log_with=["tensorboard"], project_dir=logging_dir
    )
    Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.print(objstr(config))

    accelerator.print("load model...")
    model = get_model(config)
    model = load_model(model, accelerator, config.finetune.GCM.checkpoint)

    accelerator.print("load dataset...")
    train_loader, val_loader, test_loader, example = get_dataloader(config)

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

    model, train_loader, val_loader, test_loader = accelerator.prepare(
        model, train_loader, val_loader, test_loader
    )

    # start valing
    accelerator.print("Start Valing! ")
    val_one_epoch(model, test_loader, metrics, post_trans, accelerator)
    compute_dl_score_for_example(model, config, post_trans, example)
