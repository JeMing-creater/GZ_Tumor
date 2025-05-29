from monai.networks.nets import SwinUNETR
from model.FMUNETR_class import FMUNETR as FMUNETR_class
from src.model.FMUNETR_seg import FMUNETR as FMUNETR_seg
from monai.networks.nets import resnet
from src.model.Vit import MedViT3D as ViT 
from src.model.TP_Mamba import SAM_MS


def get_model(config):
    if 'ResNet' in config.trainer.choose_model:
        # model = ResNet(in_channels=len(config.GCM_loader.checkModels), pretrained=False)
        model = resnet.resnet50(
                spatial_dims=3,
                n_input_channels=len(config.GCM_loader.checkModels),   # 对于医学图像，通常是单通道
                num_classes=1         # 类别数，可根据任务修改
            )
        print('ResNet')
    elif 'Vit' in config.trainer.choose_model:
        model = ViT(
            stem_chs=[32, 64, 128, 128],
            depths=[3, 4, 8, 3],
            path_dropout=0.2,
            num_classes=1,
            in_channels=len(config.GCM_loader.checkModels)
        )
        print('ViT')
    elif 'FMUNETR' in config.trainer.choose_model:
        if 'class' in config.trainer.choose_model:
            model = FMUNETR_class(in_chans=len(config.GCM_loader.checkModels), 
                        fussion = [1,2,4,8], 
                        kernel_sizes=[4, 2, 2, 2], 
                        depths=[1, 1, 1, 1], 
                        dims=[48, 96, 192, 384], 
                        heads=[1, 2, 4, 4], 
                        hidden_size=768, 
                        num_slices_list = [64, 32, 16, 8], 
                        out_indices=[0, 1, 2, 3])
        else:
            model = FMUNETR_seg(in_chans=len(config.GCM_loader.checkModels), 
                        out_chans=len(config.GCM_loader.checkModels),
                        fussion = [1,2,4,8], 
                        kernel_sizes=[4, 2, 2, 2], 
                        depths=[1, 1, 1, 1], 
                        dims=[48, 96, 192, 384], 
                        heads=[1, 2, 4, 4], 
                        hidden_size=768, 
                        num_slices_list = [64, 32, 16, 8], 
                        out_indices=[0, 1, 2, 3])
    elif config.trainer.choose_model == 'TP_Mamba':
        model = SAM_MS(
            in_classes = len(config.GCM_loader.checkModels),
            num_classes=2, 
            dr=16.0
        )
        print('TP_Mamba')
    else:
        model = resnet.resnet50(
                spatial_dims=3,
                n_input_channels=len(config.GCM_loader.checkModels),   # 对于医学图像，通常是单通道
                num_classes=1         # 类别数，可根据任务修改
            )
        print('ResNet')
    return model