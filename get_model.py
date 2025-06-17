from monai.networks.nets import SwinUNETR
from src.model.FMUNETR_class import FMUNETR as FMUNETR_class
from src.model.FMUNETR_seg import FMUNETR as FMUNETR_seg
from model.ResNet import resnet50
from src.model.Vit import Vit as Vit
from src.model.TP_Mamba import SAM_MS


def get_model(config):
    if "ResNet" in config.trainer.choose_model:
        # model = ResNet(in_channels=len(config.GCM_loader.checkModels), pretrained=False)
        model = resnet50(
            in_classes=len(config.GCM_loader.checkModels),
            num_classes=1,
            shortcut_type="B",
            spatial_size=64,
            sample_count=128,
        )
        print("ResNet")
    elif "Vit" in config.trainer.choose_model:
        model = Vit(
            in_channels=len(config.GCM_loader.checkModels),
            out_channels=len(config.GCM_loader.checkModels),
            embed_dim=96,
            embedding_dim=32,
            channels=(24, 48, 60),
            blocks=(1, 2, 3, 2),
            heads=(1, 2, 4, 4),
            r=(4, 2, 2, 1),
            dropout=0.3,
        )
        print("ViT")
    elif "FMUNETR" in config.trainer.choose_model:
        if config.trainer.choose_dataset == "GCM":
            use_config = config.GCM_loader
        elif config.trainer.choose_dataset == "GCNC":
            use_config = config.GCNC_loader
        if "class" in config.trainer.choose_model:
            model = FMUNETR_class(
                in_chans=len(use_config.checkModels),
                fussion=[1, 2, 4, 8],
                kernel_sizes=[4, 2, 2, 2],
                depths=[1, 1, 1, 1],
                dims=[48, 96, 192, 384],
                heads=[1, 2, 4, 4],
                hidden_size=768,
                num_slices_list=[64, 32, 16, 8],
                out_indices=[0, 1, 2, 3],
            )
        else:
            model = FMUNETR_seg(
                in_chans=len(use_config.checkModels),
                out_chans=len(use_config.checkModels),
                fussion=[1, 2, 4, 8],
                kernel_sizes=[4, 2, 2, 2],
                depths=[1, 1, 1, 1],
                dims=[48, 96, 192, 384],
                heads=[1, 2, 4, 4],
                hidden_size=768,
                num_slices_list=[64, 32, 16, 8],
                out_indices=[0, 1, 2, 3],
            )
    elif config.trainer.choose_model == "TP_Mamba":
        model = SAM_MS(
            in_classes=len(config.GCM_loader.checkModels), num_classes=2, dr=16.0
        )
        print("TP_Mamba")
    else:
        model = resnet50(
            in_classes=len(config.GCM_loader.checkModels),  # 对于医学图像，通常是单通道
            num_classes=1,
            shortcut_type="B",
            spatial_size=64,
            sample_count=128,
            # 类别数，可根据任务修改
        )
        print("ResNet")
    return model
