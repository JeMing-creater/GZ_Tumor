from monai.networks.nets import SwinUNETR
from src.model import FMUNETR, ResNet, ViT
def get_model(config):
    if config.trainer.GCM.choose_model == 'ResNet':
        model = ResNet(in_channels=len(config.GCM_loader.checkModels), pretrained=False)
        print('ResNet')
    elif config.trainer.GCM.choose_model == 'Vit':
        model = ViT(in_channels=len(config.GCM_loader.checkModels), img_size=(128, 128, 64),
                 patch_size=(16, 16, 16), embed_dim=768, depth=12, num_heads=12, mlp_dim=3072)
        print('ViT')
    elif config.trainer.GCM.choose_model == 'FMUNETR':
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
        model = ResNet(in_channels=len(config.GCM_loader.checkModels), pretrained=False)
        print('ResNet')
    return model