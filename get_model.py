from monai.networks.nets import SwinUNETR

# from src.model.Class.HWAUNETR_class import HWAUNETR as TFM_UNET_class
# from src.model.Seg.HWAUNETR_seg import HWAUNETR as TFM_UNET_seg
from src.model.Class.ResNet import resnet50
from src.model.Class.Vit import Vit as Vit
from src.model.Class.TP_Mamba import SAM_MS

# from src.model.Multi_Tasks.HSL_Net import HSL_Net


def get_model(config):
    # data choose
    if config.trainer.choose_dataset == "GCM":
        use_config = config.GCM_loader
    elif config.trainer.choose_dataset == "GCNC":
        use_config = config.GCNC_loader

    # Multitask choose model will return first
    if "HSL_Net" in config.trainer.choose_model:
        model = HSL_Net(
            in_channels=len(use_config.checkModels),
            out_channels=len(use_config.checkModels),
            num_tasks=1,
            hidden_size=768,
            depths=[2, 2, 2, 2],
            kernel_sizes=[4, 2, 2, 2],
            dims=[48, 96, 192, 384],
            out_dim=64,
            heads=[1, 2, 4, 4],
            out_indices=[0, 1, 2, 3],
            num_slices_list=[64, 32, 16, 8],
        )
        print("HSL_Net for multitask")
        return model

    # Single task choose model return now
    if config.trainer.task == "Segmentation":
        if "TFM_UNET" in config.trainer.choose_model:
            model = TFM_UNET_seg(
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
            print("TFM_UNET for segmentation")
        elif "SwinUNETR" in config.trainer.choose_model:
            model = SwinUNETR(
                in_channels=len(use_config.checkModels),
                out_channels=len(use_config.checkModels),
                img_size=use_config.target_size,
                feature_size=48,
            )
            print("SwinUNETR for segmentation")
    elif config.trainer.task == "Classification":
        if "ResNet" in config.trainer.choose_model:
            model = resnet50(
                in_classes=len(use_config.checkModels),
                num_classes=1,
                shortcut_type="B",
                spatial_size=64,
                sample_count=128,
            )
            print("ResNet for classification")

        elif "Vit" in config.trainer.choose_model:
            model = Vit(
                in_channels=len(use_config.checkModels),
                out_channels=len(use_config.checkModels),
                embed_dim=96,
                embedding_dim=32,
                channels=(24, 48, 60),
                blocks=(1, 2, 3, 2),
                heads=(1, 2, 4, 4),
                r=(4, 2, 2, 1),
                dropout=0.3,
            )
            print("ViT for classification")

        elif "TFM_UNET" in config.trainer.choose_model:
            model = TFM_UNET_class(
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
            print("TFM_UNET for classification")
        elif config.trainer.choose_model == "TP_Mamba":
            model = SAM_MS(
                in_classes=len(use_config.checkModels), num_classes=2, dr=16.0
            )
            print("TP_Mamba for classification")
    else:
        raise ValueError(
            "Invalid task type. Choose either 'segmentation' or 'classification'."
        )

    return model
