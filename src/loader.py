from calendar import c
import os
import math
import yaml
import torch
import monai
import random
import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
from datetime import datetime
from easydict import EasyDict
import torch.nn.functional as F
from monai.utils import ensure_tuple_rep
from monai.networks.utils import one_hot

sitk.ProcessObject.SetGlobalWarningDisplay(False)
from typing import Tuple, List, Mapping, Hashable, Dict
from monai.transforms import (
    LoadImaged,
    MapTransform,
    ScaleIntensityRanged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ResampleToMatchd,
    ResizeWithPadOrCropd,
    Resize,
    Resized,
    RandFlipd,
    NormalizeIntensityd,
    ToTensord,
    RandScaleIntensityd,
    RandShiftIntensityd,
)


class ConvertToMultiChannelBasedOnBratsClassesd_For_BraTS(
    monai.transforms.MapTransform
):
    """
    TC WT ET
    Dictionary-based wrapper of :py:class:`monai.transforms.ConvertToMultiChannelBasedOnBratsClasses`.
    Convert labels to multi channels based on brats18 classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    backend = [monai.utils.TransformBackends.TORCH, monai.utils.TransformBackends.NUMPY]

    def __init__(
        self,
        keys: monai.config.KeysCollection,
        is2019: bool = False,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.is2019 = is2019

    def converter(self, img: monai.config.NdarrayOrTensor):
        # TC WT ET
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)
        if self.is2019:
            result = [
                (img == 2) | (img == 3),
                (img == 1) | (img == 2) | (img == 3),
                (img == 2),
            ]
        else:
            # TC WT ET
            result = [
                (img == 1) | (img == 4),
                (img == 1) | (img == 4) | (img == 2),
                img == 4,
            ]
            # merge labels 1 (tumor non-enh) and 4 (tumor enh) and 2 (large edema) to WT
            # label 4 is ET
        return (
            torch.stack(result, dim=0)
            if isinstance(img, torch.Tensor)
            else np.stack(result, axis=0)
        )

    def __call__(
        self, data: Mapping[Hashable, monai.config.NdarrayOrTensor]
    ) -> Dict[Hashable, monai.config.NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


def read_csv_for_GCM(config):
    csv_path = config.GCM_loader.root + "/" + "Classification.xlsx"
    # 定义dtype转换，将第二列（索引为1）读作str
    dtype_converters = {1: str}

    df1 = pd.read_excel(
        csv_path, engine="openpyxl", dtype=dtype_converters, sheet_name="腹膜转移分类"
    )
    df2 = pd.read_excel(
        csv_path,
        engine="openpyxl",
        dtype=dtype_converters,
        sheet_name="淋巴结同时序（手术）",
    )
    df3 = pd.read_excel(
        csv_path,
        engine="openpyxl",
        dtype=dtype_converters,
        sheet_name="淋巴结异时序（化疗后）",
    )

    # 创建空字典
    content_dict1 = {}
    content_dict2 = {}
    content_dict3 = {}
    # 遍历DataFrame的每一行，从第二行开始
    for index, row in df1.iterrows():

        key = row[1]  # 第2列作为键
        values = row[2]  # 第3列的数据读为label
        time = row[4]  # 第4列的数据读为time，用于划分数据
        content_dict1[key] = [values, time]

    for index, row in df2.iterrows():

        key = row[1]  # 第2列作为键
        values = row[2]  # 第3列的数据读为label
        time = row[4]
        content_dict2[key] = [values, time]

    for index, row in df3.iterrows():

        key = row[1]  # 第2列作为键
        values = row[2]  # 第3列的数据读为label
        time = row[4]
        content_dict3[key] = [values, time]

    return content_dict1, content_dict2, content_dict3


def read_csv_for_GCNC(config):

    csv_path1 = config.GCNC_loader.root + "/" + "ALL.xlsx"

    # 定义dtype转换，将第四列（索引为3）读作str
    dtype_converters = {3: str}

    df = pd.read_excel(csv_path1, engine="openpyxl", dtype=dtype_converters)

    # 创建空列表
    content_dict = {}

    # 遍历DataFrame的每一行，从第二行开始
    for index, row in df.iterrows():

        key = row["病理号"]
        # TODO: 此处只做高低表达分类记录。后续数据补全可以添加阴阳性分类
        # 如果指标缺失，则跳过读取
        if isinstance(row["PD-L1"], (int, float)) != True:
            continue

        # TODO: xlxs文件内部内容修改
        if row["M分期"] == "x":
            row["M分期"] = 1
        elif row["M分期"] == 3:
            row["M分期"] = 0

        if isinstance(row["M分期"], (int, float)) != True:
            continue

        label_dict = {"PD-L1": -1, "M": -1}

        if row["PD-L1"] > 20:
            label_dict["PD-L1"] = 1
        else:
            label_dict["PD-L1"] = 0

        label_dict["M"] = row["M分期"]

        if label_dict["PD-L1"] != -1 and label_dict["M"] != -1:
            content_dict[key] = label_dict
    return content_dict


def read_usedata(file_path):
    read_flas = False
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if read_flas == True:
                result = line.replace("\n", "").split(",")
                result = [data.replace(" ", "") for data in result]
                return result
            elif "Useful data" in line:
                read_flas = True
                continue


def load_MR_dataset_images(
    root, use_data, use_models, use_data_dict={}, data_choose="GCM"
):
    images_path = os.listdir(root)
    images_list = []
    images_lack_list = []

    for path in use_data:
        if path in images_path:
            models = os.listdir(root + "/" + path + "/")
        else:
            continue
        lack_flag = False
        lack_model_flag = False
        image = []
        label = []

        for model in models:
            if "WI" in model:
                check_model = model.replace("WI", "")
            elif model == "CT1":
                check_model = "T1+C"
            elif model == "T1+c":
                check_model = "T1+C"
            else:
                check_model = model

            if check_model in use_models:
                if not os.path.exists(root + "/" + path + "/" + model):
                    print(f"{path} does not have {model} file. ")
                    lack_model_flag = True
                    break
                elif not os.path.exists(
                    root + "/" + path + "/" + model + "/" + path + ".nii.gz"
                ):
                    print(f"{path} does not have {model} image file.")
                    lack_model_flag = True
                    break

                image.append(root + "/" + path + "/" + model + "/" + path + ".nii.gz")
                if not os.path.exists(
                    root + "/" + path + "/" + model + "/" + path + "seg.nii.gz"
                ):
                    print(f"Label file not found for {path} in model {model}. ")
                    label.append(
                        root + "/" + path + "/" + model + "/" + path + ".nii.gz"
                    )
                    lack_flag = True
                else:
                    label.append(
                        root + "/" + path + "/" + model + "/" + path + "seg.nii.gz"
                    )

        if image == [] or len(image) < len(use_models):
            print(f"{path} does not have image file or not enough modals. ")
            lack_model_flag = True

        if data_choose == "GCM":
            if lack_flag == False and lack_model_flag == False:
                if use_data_dict != {}:
                    images_list.append(
                        {
                            "image": image,
                            "label": label,
                            "class_label": use_data_dict[path][0],
                        }
                    )
                else:
                    images_list.append(
                        {
                            "image": image,
                            "label": label,
                        }
                    )
            elif lack_model_flag == False:
                if use_data_dict != {}:
                    images_lack_list.append(
                        {
                            "image": image,
                            "label": label,
                            "class_label": use_data_dict[path][0],
                        }
                    )
                else:
                    images_lack_list.append(
                        {
                            "image": image,
                            "label": label,
                        }
                    )
        else:
            if lack_flag == False and lack_model_flag == False:
                if use_data_dict != {}:
                    images_list.append(
                        {
                            "image": image,
                            "label": label,
                            "pdl1_label": use_data_dict[path]["PD-L1"],
                            "m_label": use_data_dict[path]["M"],
                        }
                    )
                else:
                    images_list.append(
                        {
                            "image": image,
                            "label": label,
                        }
                    )
            elif lack_model_flag == False:
                if use_data_dict != {}:
                    images_lack_list.append(
                        {
                            "image": image,
                            "label": label,
                            "pdl1_label": use_data_dict[path]["PD-L1"],
                            "m_label": use_data_dict[path]["M"],
                        }
                    )
                else:
                    images_lack_list.append(
                        {
                            "image": image,
                            "label": label,
                        }
                    )
        # print(f'{path} example has been loaded')

    return images_list, images_lack_list


def load_brats2021_dataset_images(root):
    images_path = os.listdir(root)
    images_list = []
    for path in images_path:
        image_path = root + "/" + path + "/" + path
        flair_img = image_path + "_flair.nii.gz"
        t1_img = image_path + "_t1.nii.gz"
        t1ce_img = image_path + "_t1ce.nii.gz"
        t2_img = image_path + "_t2.nii.gz"
        seg_img = image_path + "_seg.nii.gz"
        images_list.append(
            {"image": [flair_img, t1_img, t1ce_img, t2_img], "label": seg_img}
        )
    return images_list


def get_GCM_transforms(
    config: EasyDict,
) -> Tuple[monai.transforms.Compose, monai.transforms.Compose]:
    load_transform = []
    for model_scale in config.GCM_loader.model_scale:
        load_transform.append(
            monai.transforms.Compose(
                [
                    LoadImaged(
                        keys=["image", "label"], image_only=False, simple_keys=True
                    ),
                    EnsureChannelFirstd(keys=["image", "label"]),
                    Resized(
                        keys=["image", "label"],
                        spatial_size=config.GCM_loader.target_size,
                        mode=("trilinear", "nearest-exact"),
                    ),
                    ScaleIntensityRanged(
                        keys=["image"],  # 对图像应用变换
                        a_min=model_scale[0],  # 输入图像的最小强度值
                        a_max=model_scale[1],  # 输入图像的最大强度值
                        b_min=0.0,  # 输出图像的最小强度值
                        b_max=1.0,  # 输出图像的最大强度值
                        clip=True,  # 是否裁剪超出范围的值
                    ),
                    ToTensord(keys=["image", "label"]),
                ]
            )
        )

    train_transform = monai.transforms.Compose(
        [
            # 训练集的额外增强
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = monai.transforms.Compose(
        [
            ToTensord(keys=["image", "label"]),
        ]
    )
    return load_transform, train_transform, val_transform


def get_GCNC_transforms(
    config: EasyDict,
) -> Tuple[monai.transforms.Compose, monai.transforms.Compose]:
    load_transform = []
    for model_scale in config.GCNC_loader.model_scale:
        load_transform.append(
            monai.transforms.Compose(
                [
                    LoadImaged(
                        keys=["image", "label"], image_only=False, simple_keys=True
                    ),
                    EnsureChannelFirstd(keys=["image", "label"]),
                    Resized(
                        keys=["image", "label"],
                        spatial_size=config.GCNC_loader.target_size,
                        mode=("trilinear", "nearest-exact"),
                    ),
                    ScaleIntensityRanged(
                        keys=["image"],  # 对图像应用变换
                        a_min=model_scale[0],  # 输入图像的最小强度值
                        a_max=model_scale[1],  # 输入图像的最大强度值
                        b_min=0.0,  # 输出图像的最小强度值
                        b_max=1.0,  # 输出图像的最大强度值
                        clip=True,  # 是否裁剪超出范围的值
                    ),
                    ToTensord(keys=["image", "label"]),
                ]
            )
        )

    train_transform = monai.transforms.Compose(
        [
            # 训练集的额外增强
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = monai.transforms.Compose(
        [
            ToTensord(keys=["image", "label"]),
        ]
    )
    return load_transform, train_transform, val_transform


def get_Brats_transforms(
    config: EasyDict,
) -> Tuple[monai.transforms.Compose, monai.transforms.Compose]:
    train_transform = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=["image", "label"]),
            monai.transforms.EnsureChannelFirstd(keys="image"),
            monai.transforms.EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd_For_BraTS(
                keys=["label"], is2019=False
            ),
            monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            monai.transforms.SpatialPadD(
                keys=["image", "label"],
                spatial_size=(255, 255, config.BraTS_loader.image_size),
                method="symmetric",
                mode="constant",
            ),
            monai.transforms.Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            monai.transforms.CenterSpatialCropD(
                keys=["image", "label"],
                roi_size=ensure_tuple_rep(config.BraTS_loader.image_size, 3),
            ),
            # monai.transforms.Resized(keys=["image", "label"], spatial_size=ensure_tuple_rep(config.model.image_size, 3)),
            monai.transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                num_samples=2,
                spatial_size=ensure_tuple_rep(config.BraTS_loader.image_size, 3),
                pos=1,
                neg=1,
                image_key="image",
                image_threshold=0,
            ),
            # monai.transforms.RandSpatialCropd(keys=["image", "label"], roi_size=config.model.image_size, random_size=False),
            monai.transforms.RandFlipd(
                keys=["image", "label"], prob=0.5, spatial_axis=0
            ),
            monai.transforms.RandFlipd(
                keys=["image", "label"], prob=0.5, spatial_axis=1
            ),
            monai.transforms.RandFlipd(
                keys=["image", "label"], prob=0.5, spatial_axis=2
            ),
            monai.transforms.NormalizeIntensityd(
                keys="image", nonzero=True, channel_wise=True
            ),
            monai.transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            monai.transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            monai.transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=["image", "label"]),
            monai.transforms.EnsureChannelFirstd(keys="image"),
            monai.transforms.EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd_For_BraTS(
                keys="label", is2019=False
            ),
            monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            monai.transforms.Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            # monai.transforms.Resized(keys=["image", "label"], spatial_size=ensure_tuple_rep(config.model.image_size, 3)),
            monai.transforms.NormalizeIntensityd(
                keys="image", nonzero=True, channel_wise=True
            ),
        ]
    )
    return train_transform, val_transform


class MultiModalityDataset(monai.data.Dataset):
    def __init__(
        self,
        data,
        loadforms,
        transforms,
        over_label=False,
        over_add=0,
        use_class=True,
        data_choose="GCM",
    ):
        self.data = data
        self.transforms = transforms
        self.loadforms = loadforms
        self.over_label = over_label
        self.over_add = over_add
        self.use_class = use_class
        self.data_choose = data_choose

    def extract_and_resize(self, image, label, over_add=0):
        # 获取label中值为1的点的坐标
        indices = torch.nonzero(label[0])  # 去掉batch维度，找到非零索引

        # 找到最长的维度
        max_size = max(image[0].shape[0], image[0].shape[1], image[0].shape[2])

        # 计算各个维度上的 over_add，按比例缩小较短维度的扩展量
        over_add_x = round(over_add * (image[0].shape[0] / max_size))
        over_add_y = round(over_add * (image[0].shape[1] / max_size))
        over_add_z = round(over_add * (image[0].shape[2] / max_size))

        # 获取在每个维度上的最小和最大索引
        min_x, min_y, min_z = indices.min(dim=0).values.tolist()
        max_x, max_y, max_z = indices.max(dim=0).values.tolist()

        # 计算扩展后的坐标，并限制在合法范围内
        min_x = max(0, min_x - over_add_x)
        max_x = min(image[0].shape[0] - 1, max_x + over_add_x)
        min_y = max(0, min_y - over_add_y)
        max_y = min(image[0].shape[1] - 1, max_y + over_add_y)
        min_z = max(0, min_z - over_add_z)
        max_z = min(image[0].shape[2] - 1, max_z + over_add_z)

        # 切割image和label
        cropped_image = image[
            :, min_x : max_x + 1, min_y : max_y + 1, min_z : max_z + 1
        ]

        # 直接使用 interpolate 进行 resize 到 (1, 128, 128, 64)
        # resized_image = F.interpolate(cropped_image.unsqueeze(0), size=(128, 128, 64), mode='trilinear', align_corners=False).squeeze(0)
        resized_image = monai.transforms.Resize(
            spatial_size=(label.shape[1], label.shape[2], label.shape[3]),
            mode=("trilinear"),
        )(cropped_image)
        return resized_image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        combined_data = {}

        for i in range(0, len(item["image"])):
            # print('Loading ', item['image'][i])
            globals()[f"data_{i}"] = self.loadforms[i](
                {"image": item["image"][i], "label": item["label"][i]}
            )

            combined_data[f"model_{i}_image"] = globals()[f"data_{i}"]["image"]
            combined_data[f"model_{i}_label"] = globals()[f"data_{i}"]["label"]

            if self.over_label == True:
                imgae = self.extract_and_resize(
                    combined_data[f"model_{i}_image"],
                    combined_data[f"model_{i}_label"],
                    self.over_add,
                )
            else:
                imgae = combined_data[f"model_{i}_image"]
            combined_data[f"model_{i}_image"] = imgae

        images = []
        labels = []

        for i in range(0, len(item["image"])):
            images.append(combined_data[f"model_{i}_image"])
            labels.append(combined_data[f"model_{i}_label"])
            image_tensor = torch.cat(images, dim=0)
            label_tensor = torch.cat(labels, dim=0)

        result = {"image": image_tensor, "label": label_tensor}
        result = self.transforms(result)

        if self.use_class == True:
            if self.data_choose == "GCM":
                return {
                    "image": result["image"],
                    "label": result["label"],
                    "class_label": torch.tensor(item["class_label"])
                    .unsqueeze(0)
                    .long(),
                }
            else:
                return {
                    "image": result["image"],
                    "label": result["label"],
                    "pdl1_label": torch.tensor(item["pdl1_label"]).unsqueeze(0).long(),
                    "m_label": torch.tensor(item["m_label"]).unsqueeze(0).long(),
                }
        else:
            return {"image": result["image"], "label": result["label"]}


def split_list(data, ratios):
    # 计算每个部分的大小
    sizes = [math.ceil(len(data) * r) for r in ratios]

    # 调整大小以确保总大小与原列表长度匹配
    total_size = sum(sizes)
    if total_size != len(data):
        sizes[-1] -= total_size - len(data)

    # 分割列表
    start = 0
    parts = []
    for size in sizes:
        end = start + size
        parts.append(data[start:end])
        start = end

    return parts


def check_example(data):
    index = []
    for d in data:
        num = d["image"][0].split("/")[-1].split(".")[0]
        index.append(num)
    return index


def sort_keys_by_time(data: dict) -> list:
    def parse_time(t):
        if isinstance(t, datetime):
            return t
        if isinstance(t, (float, int)):
            return datetime.fromtimestamp(t)
        if not isinstance(t, str):
            raise ValueError(f"Unsupported time format: {t}")

        t = t.strip()
        time_formats = ["%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d  %H:%M:%S"]

        for fmt in time_formats:
            try:
                return datetime.strptime(t, fmt)
            except ValueError:
                continue

        raise ValueError(f"Unrecognized datetime format: {t}")

    return sorted(data.keys(), key=lambda k: parse_time(data[k][1]))


def split_examples_to_data(data, config, lack_flag=False, loding=False):
    def read_file_to_list(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            # 去除每行末尾的换行符
        lines = [line.strip() for line in lines]
        return lines

    def select_example_to_data(data, example_list):
        selected_data = []
        for d in data:
            num = d["image"][0].split("/")[-1].split(".")[0]
            if num in example_list:
                selected_data.append(d)
        return selected_data

    def load_example_to_data(data, example_path, loding=False):
        data_list = read_file_to_list(example_path)
        print(f"Loading examples from {example_path}")
        if loding == True:
            data_list = select_example_to_data(data, data_list)
        return data_list

    train_example = config.GCM_loader.root + "/" + "train_examples.txt"
    val_example = config.GCM_loader.root + "/" + "val_examples.txt"
    test_example = config.GCM_loader.root + "/" + "test_examples.txt"

    train_data, val_data, test_data = (
        load_example_to_data(data, train_example, loding),
        load_example_to_data(data, val_example, loding),
        load_example_to_data(data, test_example, loding),
    )

    if lack_flag == True:
        train_data_lack, val_data_lack, test_data_lack = (
            load_example_to_data(data, train_example, loding),
            load_example_to_data(data, val_example, loding),
            load_example_to_data(data, test_example, loding),
        )
        return (
            train_data,
            val_data,
            test_data,
            train_data_lack,
            val_data_lack,
            test_data_lack,
        )

    return train_data, val_data, test_data


def get_dataloader_GCM(
    config: EasyDict,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    root = config.GCM_loader.root
    datapath = root + "/" + "ALL" + "/"
    use_models = config.GCM_loader.checkModels

    # data1: 腹膜转移分类; data2: 淋巴结同时序（手术）; data3: 淋巴结异时序（化疗后）
    data1, data2, data3 = read_csv_for_GCM(config)
    if config.GCM_loader.task == "PM":
        use_data_dict = data1
    elif config.GCM_loader.task == "NL_SS":
        use_data_dict = data2
    else:
        use_data_dict = data3

    # 按时间顺序划分数据集
    use_data_list = sort_keys_by_time(use_data_dict)

    # 剔除不需要的病历号
    remove_list = config.GCM_loader.leapfrog
    use_data = [item for item in use_data_list if item not in remove_list]

    # 在use_data处划分数据，避免并行导致的读取问题
    if config.GCM_loader.fix_example != True:
        if config.GCM_loader.time_limit != True:
            random.shuffle(use_data)
            print("Random Loading!")
        train_use_data, val_use_data, test_use_data = split_list(
            use_data,
            [
                config.GCM_loader.train_ratio,
                config.GCM_loader.val_ratio,
                config.GCM_loader.test_ratio,
            ],
        )
        if config.GCM_loader.fusion == True:
            need_val_data = val_use_data + test_use_data
            val_use_data = need_val_data
            test_use_data = need_val_data
    else:
        train_use_data, val_use_data, test_use_data = split_examples_to_data(
            use_data, config
        )

    # 加载MR数据
    train_data, _ = load_MR_dataset_images(
        datapath, train_use_data, use_models, use_data_dict, data_choose="GCM"
    )
    val_data, _ = load_MR_dataset_images(
        datapath, val_use_data, use_models, use_data_dict, data_choose="GCM"
    )
    test_data, _ = load_MR_dataset_images(
        datapath, test_use_data, use_models, use_data_dict, data_choose="GCM"
    )

    # data, _ = load_MR_dataset_images(datapath, use_data, use_models, use_data_dict)

    load_transform, train_transform, val_transform = get_GCM_transforms(config)

    # if config.GCM_loader.fix_example == True:
    #     train_data, val_data, test_data = split_examples_to_data(data, config)
    # else:
    #     # shuffle data for objective verification
    #     if config.GCM_loader.time_limit != True:
    #         random.shuffle(data)
    #         print('Random Loading!')

    #     train_data, val_data, test_data = split_list(data, [config.GCM_loader.train_ratio, config.GCM_loader.val_ratio, config.GCM_loader.test_ratio])

    #     # if not need test, can use fusion to fuse two data
    #     if config.GCM_loader.fusion == True:
    #         need_val_data = val_data + test_data
    #         val_data = need_val_data
    #         test_data = need_val_data

    train_example = check_example(train_data)
    val_example = check_example(val_data)
    test_example = check_example(test_data)

    train_dataset = MultiModalityDataset(
        data=train_data,
        over_label=config.GCM_loader.over_label,
        over_add=config.GCM_loader.over_add,
        loadforms=load_transform,
        transforms=train_transform,
        data_choose="GCM",
    )
    val_dataset = MultiModalityDataset(
        data=val_data,
        over_label=config.GCM_loader.over_label,
        over_add=config.GCM_loader.over_add,
        loadforms=load_transform,
        transforms=val_transform,
        data_choose="GCM",
    )
    test_dataset = MultiModalityDataset(
        data=test_data,
        over_label=config.GCM_loader.over_label,
        over_add=config.GCM_loader.over_add,
        loadforms=load_transform,
        transforms=val_transform,
        data_choose="GCM",
    )

    train_loader = monai.data.DataLoader(
        train_dataset,
        num_workers=config.GCM_loader.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=True,
    )
    val_loader = monai.data.DataLoader(
        val_dataset,
        num_workers=config.GCM_loader.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=False,
    )
    test_loader = monai.data.DataLoader(
        test_dataset,
        num_workers=config.GCM_loader.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=False,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        (train_example, val_example, test_example),
    )


def get_dataloader_GCNC(
    config: EasyDict,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    root = config.GCNC_loader.root
    datapath = root + "/" + "ALL" + "/"
    use_models = config.GCNC_loader.checkModels

    content_dict = read_csv_for_GCNC(config)

    use_data_list = list(content_dict.keys())

    remove_list = config.GCNC_loader.leapfrog
    use_data = [item for item in use_data_list if item not in remove_list]

    data, data_lack = load_MR_dataset_images(
        datapath, use_data, use_models, content_dict, data_choose="GCNC"
    )
    load_transform, train_transform, val_transform = get_GCNC_transforms(config)

    if config.GCNC_loader.fix_example == True:
        (
            train_data,
            val_data,
            test_data,
            train_data_lack,
            val_data_lack,
            test_data_lack,
        ) = split_examples_to_data(data, config, lack_flag=True, loding=True)
    else:
        random.shuffle(data)
        print("Random Loading!")
        train_data, val_data, test_data = split_list(
            data,
            [
                config.GCNC_loader.train_ratio,
                config.GCNC_loader.val_ratio,
                config.GCNC_loader.test_ratio,
            ],
        )
        train_data_lack, val_data_lack, test_data_lack = split_list(
            data_lack,
            [
                config.GCNC_loader.train_ratio,
                config.GCNC_loader.val_ratio,
                config.GCNC_loader.test_ratio,
            ],
        )

        if config.GCNC_loader.fusion == True:
            need_val_data = val_data + test_data
            val_data = need_val_data
            test_data = need_val_data

            need_val_data = val_data_lack + test_data_lack
            val_data_lack = need_val_data
            test_data_lack = need_val_data

    train_example = check_example(train_data)
    val_example = check_example(val_data)
    test_example = check_example(test_data)

    # train_lack_example = check_example(train_data_lack)
    # val_lack_example = check_example(val_data_lack)
    # test_lack_example = check_example(test_data_lack)

    train_dataset = MultiModalityDataset(
        data=train_data,
        over_label=config.GCNC_loader.over_label,
        over_add=config.GCNC_loader.over_add,
        loadforms=load_transform,
        transforms=train_transform,
        use_class=True,
        data_choose="GCNC",
    )
    val_dataset = MultiModalityDataset(
        data=val_data,
        over_label=config.GCNC_loader.over_label,
        over_add=config.GCNC_loader.over_add,
        loadforms=load_transform,
        transforms=val_transform,
        use_class=True,
        data_choose="GCNC",
    )
    test_dataset = MultiModalityDataset(
        data=test_data,
        over_label=config.GCNC_loader.over_label,
        over_add=config.GCNC_loader.over_add,
        loadforms=load_transform,
        transforms=val_transform,
        use_class=True,
        data_choose="GCNC",
    )

    # train_lack_dataset = MultiModalityDataset(data=train_data_lack, over_label=config.GCNC_loader.over_label, over_add = config.GCNC_loader.over_add,loadforms = load_transform,
    #                                      transforms=train_transform)
    # val_lack_dataset   = MultiModalityDataset(data=val_data_lack, over_label=config.GCNC_loader.over_label, over_add = config.GCNC_loader.over_add,
    #                                      loadforms = load_transform,
    #                                      transforms=val_transform)
    # test_lack_dataset   = MultiModalityDataset(data=test_data_lack, over_label=config.GCNC_loader.over_label, over_add = config.GCNC_loader.over_add,
    #                                      loadforms = load_transform,
    #                                      transforms=val_transform)

    train_loader = monai.data.DataLoader(
        train_dataset,
        num_workers=config.GCNC_loader.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=True,
    )
    val_loader = monai.data.DataLoader(
        val_dataset,
        num_workers=config.GCNC_loader.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=False,
    )
    test_loader = monai.data.DataLoader(
        test_dataset,
        num_workers=config.GCNC_loader.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=False,
    )

    # train_lack_loader = monai.data.DataLoader(train_lack_dataset, num_workers=config.GCNC_loader.num_workers,
    #                                      batch_size=config.trainer.batch_size, shuffle=True)
    # val_lack_loader = monai.data.DataLoader(val_lack_dataset, num_workers=config.GCNC_loader.num_workers,
    #                                    batch_size=config.trainer.batch_size, shuffle=False)
    # test_lack_loader = monai.data.DataLoader(test_lack_dataset, num_workers=config.GCNC_loader.num_workers,
    #                                    batch_size=config.trainer.batch_size, shuffle=False)

    return (
        train_loader,
        val_loader,
        test_loader,
        (train_example, val_example, test_example),
    )


def get_dataloader_BraTS(
    config: EasyDict,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_images = load_brats2021_dataset_images(config.BraTS_loader.dataPath)
    train_transform, val_transform = get_Brats_transforms(config)
    train_dataset = monai.data.Dataset(
        data=train_images[: int(len(train_images) * config.BraTS_loader.train_ratio)],
        transform=train_transform,
    )
    val_dataset = monai.data.Dataset(
        data=train_images[int(len(train_images) * config.BraTS_loader.train_ratio) :],
        transform=val_transform,
    )

    train_loader = monai.data.DataLoader(
        train_dataset,
        num_workers=config.BraTS_loader.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=True,
    )

    val_loader = monai.data.DataLoader(
        val_dataset,
        num_workers=config.BraTS_loader.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=False,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    config = EasyDict(
        yaml.load(
            open("/workspace/Jeming/ZT/config.yml", "r", encoding="utf-8"),
            Loader=yaml.FullLoader,
        )
    )

    # train_loader, val_loader, test_loader, _ = get_dataloader_GCM(config)
    train_loader, val_loader, test_loader, _ = get_dataloader_GCNC(config)

    for i, batch in enumerate(train_loader):
        try:
            print(batch["image"].shape)
            print(batch["label"].shape)
            print(batch["pdl1_label"].shape)
            print(batch["m_label"].shape)
        except Exception as e:
            print(f"Error occurred while loading batch {i}: {e}")
            continue

    for i, batch in enumerate(val_loader):
        try:
            print(batch["image"].shape)
            print(batch["label"].shape)
            print(batch["pdl1_label"].shape)
            print(batch["m_label"].shape)
        except Exception as e:
            print(f"Error occurred while loading batch {i}: {e}")
            continue

    for i, batch in enumerate(test_loader):
        try:
            print(batch["image"].shape)
            print(batch["label"].shape)
            print(batch["pdl1_label"].shape)
            print(batch["m_label"].shape)
        except Exception as e:
            print(f"Error occurred while loading batch {i}: {e}")
            continue
