
import einops

from monai.networks.blocks import Convolution, ResidualUnit
from monai.networks.blocks.dynunet_block import UnetResBlock

import torch
import torch.nn as nn
from torch.nn import init
import timm
from timm.layers import resample_abs_pos_embed_nhwc

import math

def desequence(x):
    return einops.rearrange(x, 'b c h w d -> (b d) c h w')
def sequence(x, batch, depth):
    return einops.rearrange(x, '(b d) c h w -> b c h w d', b=batch, d=depth)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

act_params = ("gelu")
class Conv3D_UP(nn.Module):
    def __init__(self, in_size, out_size, scale=(2, 2, 1), kernel_size=(1, 1, 3), padding_size=(0, 0, 1), init_stride=(1, 1, 1)):
        super(Conv3D_UP, self).__init__()
        self.convup = nn.Sequential(
                                    Convolution(3, in_size, out_size, strides=scale, kernel_size=kernel_size, act=act_params, is_transposed=True, adn_ordering='NDA'),
                                    ResidualUnit(3, out_size, out_size, strides=1, kernel_size=kernel_size, subunits=1, act=act_params, adn_ordering='NDA')
                                    )

    def forward(self, inputs):
        outputs = self.convup(inputs)
        return outputs

# class SAM_Decoder(nn.Module):
#     def __init__(self, in_dim, out_dim, num_classes=1, downsample_rate=8.0):
#         super(SAM_Decoder, self).__init__()

#         #### 2x upsample features from the last four blocks in SAM_ViT
#         self.up4 = Conv3D_UP(in_dim, out_dim, (2, 2, 1), 3, 1)
#         self.up3 = Conv3D_UP(in_dim, out_dim, (2, 2, 1), 3, 1)
#         self.up2 = Conv3D_UP(in_dim, out_dim, (2, 2, 1), 3, 1)
#         self.up1 = Conv3D_UP(in_dim, out_dim, (2, 2, 1), 3, 1)

#         #### upsample gathered features back to original resolution
#         num_upsample = int(math.log2(downsample_rate) - 1)
#         layers = [UnetResBlock(3, out_dim * 4, out_dim, 3, stride=1, act_name=act_params, norm_name="instance")] + [
#             Conv3D_UP(out_dim, out_dim, (2, 2, 1), 3, 1) for i in range(num_upsample)]

#         self.up = nn.Sequential(*layers)

#         self.final = nn.Conv3d(out_dim, num_classes, 1)

#     def forward(self, x_lis, batch, depth):
#         x4 = sequence(x_lis[-1], batch, depth)
#         x4 = self.up4(x4)
#         x3 = sequence(x_lis[-2], batch, depth)
#         x3 = self.up3(x3)
#         x2 = sequence(x_lis[-3], batch, depth)
#         x2 = self.up2(x2)
#         x1 = sequence(x_lis[-4], batch, depth)
#         x1 = self.up1(x1)

#         x = torch.cat([x1, x2, x3, x4], dim=1)
#         x = self.up(x)

#         output = self.final(x)

#         return output

class SAM_Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, downsample_rate=8.0):
        super(SAM_Decoder, self).__init__()

        # 多尺度特征上采样
        self.up4 = Conv3D_UP(in_dim, out_dim, (2, 2, 1), 3, 1)
        self.up3 = Conv3D_UP(in_dim, out_dim, (2, 2, 1), 3, 1)
        self.up2 = Conv3D_UP(in_dim, out_dim, (2, 2, 1), 3, 1)
        self.up1 = Conv3D_UP(in_dim, out_dim, (2, 2, 1), 3, 1)

        # 进一步融合
        num_upsample = int(math.log2(downsample_rate) - 1)
        layers = [
            UnetResBlock(3, out_dim * 4, out_dim, 3, stride=1, act_name=act_params, norm_name="instance")
        ] + [Conv3D_UP(out_dim, out_dim, (2, 2, 1), 3, 1) for _ in range(num_upsample)]
        self.up = nn.Sequential(*layers)

        # 全局池化 + 分类器
        self.pool = nn.AdaptiveAvgPool3d(1)  # 输出 (B, C, 1, 1, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),                    # (B, C)
            nn.Linear(out_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)                # 二分类 logits
        )

    def forward(self, x_lis, batch, depth):
        x4 = sequence(x_lis[-1], batch, depth)
        x4 = self.up4(x4)
        x3 = sequence(x_lis[-2], batch, depth)
        x3 = self.up3(x3)
        x2 = sequence(x_lis[-3], batch, depth)
        x2 = self.up2(x2)
        x1 = sequence(x_lis[-4], batch, depth)
        x1 = self.up1(x1)

        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.up(x)

        x = self.pool(x)             # (B, C, 1, 1, 1)
        out = self.classifier(x)     # (B, 1)

        return out

act_func = nn.GELU()

class CIR(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel, padding, dilation=1,groups=1):
        super(CIR, self).__init__(
            nn.Conv3d(in_channels, out_channels, kernel, padding=padding, dilation=dilation, groups=groups, bias=False),
            nn.InstanceNorm3d(out_channels),
            act_func,
        )

class Adapter_MSConv(nn.Module):
    def __init__(self, kernel=3, dim=768):
        super().__init__()
        self.dim = dim
        r = self.dim // 4
        ratio = kernel // 2
        dilation = [1, 2, 4, 8]

        self.down = CIR(self.dim, r, (1, 1, kernel), (0, 0, ratio))

        self.b1 = CIR(r, r // 4, (1, 1, kernel), (0, 0, dilation[0] * ratio), dilation[0])
        self.b2 = CIR(r, r // 4, (1, 1, kernel), (0, 0, dilation[1] * ratio), dilation[1])
        self.b3 = CIR(r, r // 4, (1, 1, kernel), (0, 0, dilation[2] * ratio), dilation[2])
        self.b4 = CIR(r, r // 4, (1, 1, kernel), (0, 0, dilation[3] * ratio), dilation[3])

        self.up = nn.Conv3d(r, self.dim, (1, 1, kernel), padding=(0, 0, ratio), bias=False)

        self.down.apply(weights_init_kaiming)
        self.b1.apply(weights_init_kaiming)
        self.b2.apply(weights_init_kaiming)
        self.b3.apply(weights_init_kaiming)
        self.b4.apply(weights_init_kaiming)
        nn.init.zeros_(self.up.weight)

    def forward(self, x, b=1, d=96):
        shortcut = x
        x = einops.rearrange(x, '(b d) h w c -> b c h w d', b=b, d=d)
        x = self.down(x)
        x = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        x = self.up(x)
        x = einops.rearrange(x, 'b c h w d -> (b d) h w c')
        x = shortcut + x
        return x

class _LoRA_qkv_timm(nn.Module):
    def __init__(
        self,
        qkv: nn.Module,
        r: int
    ):
        super().__init__()
        self.r = r
        self.qkv = qkv
        self.dim = qkv.in_features
        self.linear_a_q = nn.Linear(self.dim, r, bias=False)
        self.linear_b_q = nn.Linear(r, self.dim, bias=False)
        self.linear_a_v = nn.Linear(self.dim, r, bias=False)
        self.linear_b_v = nn.Linear(r, self.dim, bias=False)
        self.act = act_func
        self.w_identity = torch.eye(self.dim)
        self.reset_parameters()
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.linear_a_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.linear_a_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.linear_b_q.weight)
        nn.init.zeros_(self.linear_b_v.weight)
    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.act(self.linear_a_q(x)))
        new_v = self.linear_b_v(self.act(self.linear_a_v(x)))
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv

class SAM_MS(nn.Module):
    def __init__(self, model_type='samvit_base_patch16.sa1b', in_classes =2, num_classes=1, dr=8.0):
        super().__init__()
        droppath = 0.0
        model = timm.create_model(
            model_type,
            pretrained=False,  # True
            num_classes=num_classes,
        )
        for name, param in model.named_parameters():
            param.requires_grad = False

        ##### get feature maps from the last four blocks.
        self.begin = 8
        self.len = 12

        ##### Begin inserting LoRA into MHSA module
        for t_layer_i, blk in enumerate(model.blocks):
            if t_layer_i < self.len:
                model.blocks[t_layer_i].attn.qkv = _LoRA_qkv_timm(blk.attn.qkv, 64)

        ##### Begin inserting block-wise adapters, multi-scale convolutional adapter.
        self.Adapter = nn.Sequential(*[Adapter_MSConv(kernel=3) for i in range(self.len)])

        model.neck = nn.Identity()  # remove original decoders
        self.sam = model
        self.sam.patch_embed.proj = nn.Conv2d(
            in_channels=in_classes,
            out_channels=768,  # output channels for SAM
            kernel_size=(16, 16),
            stride=(16, 16),
        )
        del model

        #### define the decoder
        self.num_classes = num_classes
        self.decoder = SAM_Decoder(768, 64, downsample_rate=dr)

    def forward_ppn(self, x):
        '''
        forward for the patch embedding, position embedding and pre-norm
        '''
        self.batch, self.depth = x.shape[0], x.shape[-1]
        x = x.expand(-1, 2, -1, -1, -1)
        x = desequence(x)
        x = self.sam.patch_embed(x)
        if self.sam.pos_embed is not None:
            x = x + resample_abs_pos_embed_nhwc(self.sam.pos_embed, x.shape[1:3])
        x = self.sam.pos_drop(x)
        x = self.sam.patch_drop(x)
        x = self.sam.norm_pre(x)
        return x

    def forward_encoder(self, x):
        res = []
        for i in range(self.len):
            x = self.sam.blocks[i](x)  ### block with LoRA
            x = self.Adapter[i](x, self.batch, self.depth)  ### MS conv
            if i >= self.begin:
                res.append(x.permute(0, 3, 1, 2))
        return res

    def forward_decoder(self, x_lis):
        output = self.decoder(x_lis, self.batch, self.depth)
        return output

    def forward(self, x):
        x = self.forward_ppn(x)
        features = self.forward_encoder(x)
        output = self.forward_decoder(features)
        return output

if __name__ == '__main__':
    sam_ms = SAM_MS(in_classes =2, num_classes=2, dr=16.0)
    # x = torch.randn(1, 1, 96, 96, 96) # batch, channel (default 1 for CT), Height, Width, Depth
    x = torch.randn(2, 2, 128, 128, 64)
    y = sam_ms(x)
    print(y.size())