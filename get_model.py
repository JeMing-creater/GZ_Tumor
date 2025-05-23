from monai.networks.nets import SwinUNETR
from src.model.FMUNETR import FMUNETR
from monai.networks.nets import resnet
from src.model.Vit import ViT3D as ViT 

def get_model(config):
    if config.trainer.choose_model == 'ResNet':
        # model = ResNet(in_channels=len(config.GCM_loader.checkModels), pretrained=False)
        model = resnet.resnet50(
                spatial_dims=3,
                n_input_channels=len(config.GCM_loader.checkModels),   # 对于医学图像，通常是单通道
                num_classes=1         # 类别数，可根据任务修改
            )
        print('ResNet')
    elif config.trainer.choose_model == 'Vit':
        model = ViT(
            in_channels=len(config.GCM_loader.checkModels), img_size=(128, 128, 64)
        )
        print('ViT')
    elif config.trainer.choose_model == 'FMUNETR':
        FMUNETR = model = FMUNETR(in_chans=len(config.GCM_loader.checkModels), 
                     fussion = [1,2,4,8], 
                     kernel_sizes=[4, 2, 2, 2], 
                     depths=[1, 1, 1, 1], 
                     dims=[48, 96, 192, 384], 
                     heads=[1, 2, 4, 4], 
                     hidden_size=768, 
                     num_slices_list = [64, 32, 16, 8], 
                     out_indices=[0, 1, 2, 3])
        print('FMUNETR')
    else:
        model = resnet.resnet50(
                spatial_dims=3,
                n_input_channels=len(config.GCM_loader.checkModels),   # 对于医学图像，通常是单通道
                num_classes=1         # 类别数，可根据任务修改
            )
        print('ResNet')
    return model
