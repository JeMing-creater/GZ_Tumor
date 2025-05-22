from monai.networks.nets import SwinUNETR
from src.model.FMUNETR import FMUNETR
from monai.networks.nets import resnet
from monai.networks.nets import ViT
def get_model(config):
    if config.trainer.choose_model == 'ResNet':
        # model = ResNet(in_channels=len(config.GCM_loader.checkModels), pretrained=False)
        model = resnet.resnet50(
                spatial_dims=3,
                n_input_channels=len(config.GCM_loader.checkModels),   # 对于医学图像，通常是单通道
                num_classes=1         # 类别数，可根据任务修改
            )
        print('ResNet')
    elif config.trainer.trainer.choose_model == 'Vit':
        model = ViT(
            in_channels=len(config.GCM_loader.checkModels),            # 输入通道数：医学图像通常为1
            img_size=(64, 128, 128),  # 输入图像大小 (D, H, W)
            patch_size=(16, 16, 16),  # patch 尺寸
            pos_embed='perceptron',   # 位置编码方法：'perceptron'适用于3D
            classification=True,      # 启用分类任务
            num_classes=1,            # 分类类别数
            hidden_size=768,          # Transformer 隐藏层维度
            mlp_dim=3072,             # MLP 中间层维度
            num_layers=12,            # Transformer 层数
            num_heads=12,             # 注意力头数
            dropout_rate=0.0
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
