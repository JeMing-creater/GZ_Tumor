import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import sys
import csv
import cv2
from datetime import datetime
from typing import Dict
from copy import deepcopy
import monai
import torch
import yaml
import math
import nibabel
import scipy.ndimage as ndimage
from skimage.transform import resize
import pandas as pd
import numpy as np
import torch.nn as nn
from dataclasses import dataclass, field
from accelerate import Accelerator
from easydict import EasyDict
from monai.utils import ensure_tuple_rep
from objprint import objstr
from timm.optim import optim_factory
import torch.nn.functional as F
from matplotlib import cm
import matplotlib.pyplot as plt
from src import utils
from src.loader import get_dataloader_GCM as get_dataloader
from src.loader import get_GCM_transforms as get_transforms
from src.utils import Logger, write_example, resume_train_state, split_metrics, load_model_dict, ensure_directory_exists, copy_file
import nibabel as nib
from src.eval import calculate_f1_score, specificity, quadratic_weighted_kappa, top_k_accuracy, calculate_metrics, accumulate_metrics, compute_final_metrics

from get_model import get_model


class LayerActivations:
    features = None
 
    def __init__(self, model):
        self.hook = model.register_forward_hook(self.hook_fn)
        # 获取model.features中某一层的output
 
    def hook_fn(self, module, MRI_tensorut, output):
        self.features = output.cpu()
 
    def remove(self):  ## remove hook
        self.hook.remove()


def load_model(model, accelerator, checkpoint):
    try:
        check_path = f"{os.getcwd()}/model_store/{checkpoint}/best/"
        accelerator.print('load model from %s' % check_path)
        checkpoint = load_model_dict(check_path + 'pytorch_model.bin', )
        model.load_state_dict(checkpoint)
        accelerator.print(f'Load checkpoint model successfully!')
    except Exception as e:
        accelerator.print(e)
        accelerator.print(f'Failed to load checkpoint model!')
    return model


def write_heatmap(config, model, accelerator):
    # 验证
    model.eval()
    load_transform, _, _ = get_transforms(config)
    # cam = GradCAM(nn_module=model, target_layers=config.visualization.heatmap.target_layers)
    
    choose_image = config.GCM_loader.root + '/' + 'ALL' + '/' + f'{config.visualization.heatmap.choose_image}'
    accelerator.print('valing heatmap for image: ', choose_image)
    images = []
    labels = []
    image_size = []
    affines = []
    MRI_array_list = []
    for i in range(len(config.GCM_loader.checkModels)):
        image_path = choose_image + '/' + config.GCM_loader.checkModels[i] + '/' + f'{config.visualization.heatmap.choose_image}.nii.gz'
        label_path = choose_image + '/' + config.GCM_loader.checkModels[i] + '/' + f'{config.visualization.heatmap.choose_image}seg.nii.gz'

        MRI = nibabel.load(image_path)
        MRI_array = MRI.get_fdata()
        MRI_array = MRI_array.astype('float32')
        MRI_array_list.append(MRI_array)

        batch = load_transform[i]({
            'image': image_path,
            'label': label_path
        })
        images.append(batch['image'].unsqueeze(1))
        labels.append(batch['label'].unsqueeze(1))
        image_size.append(tuple(batch['image_meta_dict']['spatial_shape'][i].item() for i in range(3)))
        affines.append(batch['label_meta_dict']['affine'])

    input_data = torch.cat(images, dim=1).to(accelerator.device)
    conv_out = LayerActivations(model.hidden_downsample)

    _ = model(input_data)
    cam = conv_out.features
    conv_out.remove  # delete the hook

    print('cam.shape1', cam.shape)
    cam = cam.cpu().detach().numpy().squeeze()
    print('cam.shape2', cam.shape)
    cam = cam[1]
    print('cam.shape3', cam.shape)

    count = 1
    for MRI_array in MRI_array_list:

        capi = resize(cam, (MRI_array.shape[0], MRI_array.shape[1], MRI_array.shape[2]))
        capi = np.maximum(capi, 0)
        heatmap = (capi - capi.min()) / (capi.max() - capi.min())
        f, axarr = plt.subplots(3, 3, figsize=(12, 12))
        
        f.suptitle('CAM_3D_medical_image', fontsize=30)

        # 取中间值作为切片
        axial_slice_count = math.ceil(MRI_array.shape[-1]/2)
        coronal_slice_count = math.ceil(MRI_array.shape[-2]/2)
        sagittal_slice_count = math.ceil(MRI_array.shape[-3]/2)

        sagittal_MRI_img = np.squeeze(MRI_array[sagittal_slice_count, :, :])
        sagittal_grad_cmap_img = np.squeeze(heatmap[sagittal_slice_count, :, :])
        
        axial_MRI_img = np.squeeze(MRI_array[:, :, axial_slice_count])
        axial_grad_cmap_img = np.squeeze(heatmap[:, :, axial_slice_count])
        
        coronal_MRI_img = np.squeeze(MRI_array[:, coronal_slice_count, :])
        coronal_grad_cmap_img = np.squeeze(heatmap[:, coronal_slice_count, :])
        
        # Sagittal view
        img_plot = axarr[0, 0].imshow(np.rot90(sagittal_MRI_img, 1), cmap='gray')
        axarr[0, 0].axis('off')
        axarr[0, 0].set_title('Sagittal MRI', fontsize=25)
        
        img_plot = axarr[0, 1].imshow(np.rot90(sagittal_grad_cmap_img, 1), cmap='jet')
        axarr[0, 1].axis('off')
        axarr[0, 1].set_title('Weight-CAM', fontsize=25)

        # Zoom in ten times to make the weight map smoother
        sagittal_MRI_img = ndimage.zoom(sagittal_MRI_img, (1, 1), order=3)
        # Overlay the weight map with the original image
        sagittal_overlay = cv2.addWeighted(sagittal_MRI_img, 0.3, sagittal_grad_cmap_img, 0.6, 0)
        
        img_plot = axarr[0, 2].imshow(np.rot90(sagittal_overlay, 1), cmap='jet')
        axarr[0, 2].axis('off')
        axarr[0, 2].set_title('Overlay', fontsize=25)
        
        # Axial view
        img_plot = axarr[1, 0].imshow(np.rot90(axial_MRI_img, 1), cmap='gray')
        axarr[1, 0].axis('off')
        axarr[1, 0].set_title('Axial MRI', fontsize=25)

        img_plot = axarr[1, 1].imshow(np.rot90(axial_grad_cmap_img, 1), cmap='jet')
        axarr[1, 1].axis('off')
        axarr[1, 1].set_title('Weight-CAM', fontsize=25)
        
        axial_MRI_img = ndimage.zoom(axial_MRI_img, (1, 1), order=3)
        axial_overlay = cv2.addWeighted(axial_MRI_img, 0.3, axial_grad_cmap_img, 0.6, 0)
        
        img_plot = axarr[1, 2].imshow(np.rot90(axial_overlay, 1), cmap='jet')
        axarr[1, 2].axis('off')
        axarr[1, 2].set_title('Overlay', fontsize=25)
        
        # coronal view
        img_plot = axarr[2, 0].imshow(np.rot90(coronal_MRI_img, 1), cmap='gray')
        axarr[2, 0].axis('off')
        axarr[2, 0].set_title('Coronal MRI', fontsize=50)
        
        img_plot = axarr[2, 1].imshow(np.rot90(coronal_grad_cmap_img, 1), cmap='jet')
        axarr[2, 1].axis('off')
        axarr[2, 1].set_title('Weight-CAM', fontsize=50)
        
        coronal_ct_img = ndimage.zoom(coronal_MRI_img, (1, 1), order=3)
        Coronal_overlay = cv2.addWeighted(coronal_ct_img, 0.3, coronal_grad_cmap_img, 0.6, 0)
        
        img_plot = axarr[2, 2].imshow(np.rot90(Coronal_overlay, 1), cmap='jet')
        axarr[2, 2].axis('off')
        axarr[2, 2].set_title('Overlay', fontsize=50)
        
        plt.colorbar(img_plot,shrink=0.5) # color bar if need
        # plt.show()
        ensure_directory_exists(config.visualization.heatmap.write_path)
        plt.savefig(config.visualization.heatmap.write_path + '/' + f'CAM_demo_test_{count}.png')
        count += 1



if __name__ == '__main__':
    config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    utils.same_seeds(50)
    logging_dir = os.getcwd() + '/logs/' + config.finetune.checkpoint + str(datetime.now()).replace(' ','_').replace('-','_').replace(':','_').replace('.','_')
    accelerator = Accelerator(cpu=False, log_with=["tensorboard"], logging_dir=logging_dir)
    Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.print(objstr(config))
    
    accelerator.print('load model...')

    model = get_model(config)
    model = load_model(model, accelerator, config.finetune.checkpoint)
    model= accelerator.prepare(model)
    
    accelerator.print('write heatmap...')
    write_heatmap(config, model, accelerator)