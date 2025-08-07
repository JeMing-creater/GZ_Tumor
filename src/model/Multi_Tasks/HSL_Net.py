import os
import math
from cv2 import CLAHE
import torch
import warnings
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from mamba_ssm import Mamba
from timm.models.registry import register_model
from timm.models.layers import DropPath, to_2tuple, make_divisible, trunc_normal_

warnings.filterwarnings("ignore")


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i * torch.sigmoid(i)

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_i = torch.sigmoid(ctx.saved_variables[0])
        return grad_output * (
            sigmoid_i * (1 + ctx.saved_variables[0] * (1 - sigmoid_i))
        )


# Swish激活函数
class Swish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Mlp(nn.Module):
    def __init__(self, hidden_size, mlp_dim, shallow=True):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        if shallow == True:
            self.act = nn.GELU()
        else:
            self.act = Swish()
        # self.act = nn.ReLU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class HSL_Block(nn.Module):
    def __init__(self, dim, out_dim=0, num_slices=4, shallow=True):
        super().__init__()

        self.proj_1 = nn.Conv3d(dim, dim, 1)
        self.act = nn.GELU()

        if out_dim != 0:
            self.spatial_gating_unit = HSL_C(
                in_dim=dim,
                out_dim=out_dim,
                shallow=shallow,
                num_slices=num_slices,
            )
            dim = out_dim
        else:
            self.spatial_gating_unit = HSL_C(
                in_dim=dim,
                out_dim=dim,
                shallow=shallow,
                num_slices=num_slices,
            )
            dim = dim

        self.proj_2 = nn.Conv3d(dim, dim, 1)

        self.mlp = Mlp(dim, dim * 2, shallow)
        self.out_dim = out_dim

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.act(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        if self.out_dim == 0:
            x = x + shortcut
        x = self.mlp(x)
        return x


class GMPBlock(nn.Module):
    def __init__(self, in_channles, shallow=True) -> None:
        super().__init__()

        self.proj = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channles)
        if shallow == True:
            self.nonliner = nn.GELU()
        else:
            self.nonliner = Swish()
        # self.nonliner = nn.ReLU()

        self.proj2 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channles)
        if shallow == True:
            self.nonliner2 = nn.GELU()
        else:
            self.nonliner2 = Swish()

        self.proj3 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channles)
        if shallow == True:
            self.nonliner3 = nn.GELU()
        else:
            self.nonliner3 = Swish()

        self.proj4 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channles)
        if shallow == True:
            self.nonliner4 = nn.GELU()
        else:
            self.nonliner4 = Swish()

    def forward(self, x):

        x_residual = x

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)

        return x + x_residual


class Encoder(nn.Module):
    def __init__(
        self,
        in_chans=4,
        kernel_sizes=[4, 2, 2, 2],
        depths=[1, 1, 1, 1],
        dims=[48, 96, 192, 384],
        num_slices_list=[64, 32, 16, 8],
        hidden_size=768,
        out_indices=[0, 1, 2, 3],
        heads=[1, 2, 4, 4],
    ):
        super().__init__()
        self.downsample_layers = (
            nn.ModuleList()
        )  # stem and 3 intermediate downsampling conv layers
        self.dims = dims
        stem = nn.Sequential(
            nn.Conv3d(
                in_chans, dims[0], kernel_size=kernel_sizes[0], stride=kernel_sizes[0]
            ),
        )

        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(
                    dims[i],
                    dims[i + 1],
                    kernel_size=kernel_sizes[i + 1],
                    stride=kernel_sizes[i + 1],
                ),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        cur = 0
        for i in range(4):
            shallow = True
            if i > 1:
                shallow = False
            gsc = GMPBlock(dims[i], shallow)

            stage = nn.Sequential(
                *[
                    HSL_Block(
                        dim=dims[i],
                        num_slices=num_slices_list[i],
                        shallow=(True if i <= 1 else False),
                    )
                    for j in range(depths[i])
                ]
            )

            self.stages.append(stage)

            self.gscs.append(gsc)
            cur += depths[i]

        self.out_indices = out_indices

        self.hidden_downsample = nn.Conv3d(
            dims[3], hidden_size, kernel_size=2, stride=2
        )

        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)
            if i_layer >= 2:
                self.mlps.append(Mlp(dims[i_layer], 2 * dims[i_layer], False))
            else:
                self.mlps.append(Mlp(dims[i_layer], 2 * dims[i_layer], True))

    def forward(self, x):
        # outs = []
        feature_out = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.gscs[i](x)
            # x = self.stages[i](x)
            feature_out.append(self.stages[i](x))
            # feature_out.append(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x = norm_layer(x)
                x = self.mlps[i](x)
                # outs.append(x_out)
        x = self.hidden_downsample(x)
        return x, feature_out


class Convblock(nn.Module):
    def __init__(self, input_dim, dim, shallow=False):
        super().__init__()
        if shallow == True:
            self.act = nn.GELU()
        else:
            self.act = Swish()
        self.conv1 = nn.Sequential(
            nn.Conv3d(input_dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(dim),
            self.act,
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(dim),
            self.act,
        )

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        return output


class HSL(nn.Module):
    def __init__(
        self,
        in_dim=1,
        out_dim=32,
        d_state=16,
        d_conv=4,
        expand=2,
        head=4,
        num_slices=4,
        step=1,
        goble=True,
    ):
        super(HSL, self).__init__()

        # 大核全局扫描
        if goble == True:
            # 大核全局扫描
            self.att_conv = nn.Conv3d(
                in_dim,
                in_dim,
                kernel_size=7,
                stride=1,
                padding=9,
                groups=in_dim,
                dilation=3,
            )
        else:
            # 小核局部扫描
            self.att_conv = nn.Conv3d(
                in_dim, in_dim, kernel_size=5, stride=1, padding=2, groups=in_dim
            )

        self.spatial_se = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=2, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

        # Mamba
        self.norm = nn.LayerNorm(in_dim)
        self.mamba = Mamba(
            d_model=in_dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            bimamba_type="v2",  # TODO: set 154 assert bimamba_type=="v3" as none
            nslices=num_slices,
        )
        # none requires_grad
        self.mamba.A_s_log = None
        self.mamba.D_s = None
        self.mamba.conv1d_s = None
        self.mamba.x_proj_s = None
        self.mamba.dt_proj_s = None

        # 调整通道Conv
        # self.conv = nn.Conv3d(in_dim, out_dim, 3, 1, 1)

    def forward(self, x):

        # 保持原始x
        att1 = x

        # 全局特征提取
        x = self.att_conv(x)

        # 维度记录
        B, C, H, W, D = x.shape

        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]

        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        # x_mamba, o_1, o_2, o_3 = self.mamba(x_norm)
        out, q, k, v = self.mamba(x_norm)

        # 维度转换
        att2 = out.transpose(-1, -2).reshape(B, C, *img_dims)

        # 调整通道
        att = torch.cat([att1, att2], dim=1)
        avg_att = torch.mean(att, dim=1, keepdim=True)
        max_att, _ = torch.max(att, dim=1, keepdim=True)
        att = torch.cat([avg_att, max_att], dim=1)
        att = self.spatial_se(att)
        output = att1 * att[:, 0, :, :, :].unsqueeze(1) + att2 * att[
            :, 1, :, :, :
        ].unsqueeze(1)

        # x = self.conv(output)
        x = output

        return x, q, k, v


class HSL_C(nn.Module):
    def __init__(self, in_dim=3, out_dim=32, num_slices=4, shallow=True):
        super(HSL_C, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.gobel_attention = HSL(
            in_dim=in_dim // 2, out_dim=out_dim, num_slices=num_slices, goble=True
        )
        self.local_attention = HSL(
            in_dim=in_dim // 2, out_dim=out_dim, num_slices=num_slices, goble=False
        )
        self.downsample = Convblock(in_dim * 2, out_dim, shallow=shallow)

    def auto_shape(self, N):
        cube_root = round(N ** (1 / 3))
        for d in range(cube_root, 0, -1):
            if N % d != 0:
                continue
            rest = N // d
            for h in range(int(math.sqrt(rest)), 0, -1):
                if rest % h == 0:
                    w = rest // h
                    return (d, h, w)

    def forward(self, x):
        x_0, x_1 = x.chunk(2, dim=1)
        x_0, _, k, v = self.gobel_attention(x_0)
        x_1, _, q, _ = self.local_attention(x_1)

        B, C, H, W, D = x.shape
        

        q, k, v = q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)
        attn = (q.transpose(-2, -1) @ k).softmax(-1)
        out_a = v @ attn.transpose(-2, -1)

        out_a = out_a.view(B, -1, H, W, D)

        x_f = torch.cat([x_0, x_1], dim=1)

        x = torch.cat([x_f, out_a], dim=1)

        x = self.downsample(x)
        return x


class TransposedConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out, r, cat_flag=False):
        super(TransposedConvLayer, self).__init__()
        self.transposed1 = nn.ConvTranspose3d(dim_in, dim_out, kernel_size=r, stride=r)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim_out)
        if cat_flag == True:
            self.transposed2 = nn.ConvTranspose3d(
                dim_out * 2, dim_out, kernel_size=1, stride=1
            )

    def forward(self, x, feature=None):
        x = self.transposed1(x)
        # x = torch.cat((x, feature), dim=1)
        if feature != None:
            x = torch.cat((x, feature), dim=1)
            x = self.transposed2(x)
        x = self.norm(x)
        return x


class Class_Decoder(nn.Module):
    def __init__(
        self,
        task=1,
        dims=[48, 96, 192, 384],
        kernel_sizes=[4, 2, 2, 2],
        num_slices_list=[64, 32, 16, 8],
    ):
        super().__init__()

        self.fu1 = TransposedConvLayer(
            dim_in=dims[3], dim_out=dims[2], r=kernel_sizes[3]
        )
        self.block1 = HSL_Block(
            dim=dims[2] * 2,
            out_dim=dims[2],
            num_slices=num_slices_list[2],
            shallow=False,
        )
        self.ups = TransposedConvLayer(
            dim_in=dims[2], dim_out=dims[0], r=kernel_sizes[2] * 2
        )

        self.fu2 = TransposedConvLayer(
            dim_in=dims[1], dim_out=dims[0], r=kernel_sizes[1]
        )

        self.block2 = HSL_Block(
            dim=dims[0] * 2,
            out_dim=dims[0],
            num_slices=num_slices_list[0],
            shallow=True,
        )

        self.block3 = HSL_Block(
            dim=dims[0] * 2,
            out_dim=dims[0],
            num_slices=num_slices_list[0],
            shallow=True,
        )

        # 添加全局平均池化层
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.task_head = nn.Sequential(
            nn.Linear(dims[0] , dims[0] * 2),
            nn.ReLU(),
            nn.Linear(dims[0] * 2, task),  # 全连接层  # 输出层
        )


    def forward(self, feature_out):
        c1, c2, c3, c4 = feature_out

        c4 = self.fu1(c4)
        fuse_1 = torch.cat([c3, c4], dim=1)
        fuse_1 = self.block1(fuse_1)
        fuse_1 = self.ups(fuse_1)

        c2 = self.fu2(c2)
        fuse_2 = torch.cat([c2, c1], dim=1)
        fuse_2 = self.block2(fuse_2)
        x = torch.cat([fuse_1, fuse_2], dim=1)

        x = self.block3(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.task_head(x)
        # x = self.SegHead(x)
        return x


class Seg_Decoder(nn.Module):

    def __init__(self, in_channels, out_chans, dims, kernel_sizes=[4, 2, 2, 2]):
        super().__init__()

        self.TSconv1 = TransposedConvLayer(
            dim_in=in_channels, dim_out=dims[3], r=2, cat_flag=True
        )

        self.TSconv2 = TransposedConvLayer(
            dim_in=dims[3], dim_out=dims[2], r=kernel_sizes[3], cat_flag=True
        )
        self.TSconv3 = TransposedConvLayer(
            dim_in=dims[2], dim_out=dims[1], r=kernel_sizes[2], cat_flag=True
        )
        self.TSconv4 = TransposedConvLayer(
            dim_in=dims[1], dim_out=dims[0], r=kernel_sizes[1], cat_flag=True
        )

        self.SegHead = nn.ConvTranspose3d(
            dims[0], out_chans, kernel_size=kernel_sizes[0], stride=kernel_sizes[0]
        )

    def forward(self, hidden_downsample, feature_out):

        # down1, down2, down3, down4 = feature_out

        x = self.TSconv1(hidden_downsample, feature_out[-1])
        x = self.TSconv2(x, feature_out[-2])
        x = self.TSconv3(x, feature_out[-3])
        x = self.TSconv4(x, feature_out[-4])
        x = self.SegHead(x)

        return x


class HSL_Net(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        num_tasks=1,
        hidden_size=768,
        depths=[2, 2, 2, 2],
        kernel_sizes=[4, 2, 2, 2],
        dims=[48, 96, 192, 384],
        out_dim=64,
        heads=[1, 2, 4, 4],
        out_indices=[0, 1, 2, 3],
        num_slices_list=[64, 32, 16, 8]
    ):
        super().__init__()

        self.encoder = Encoder(
            in_chans=in_channels,
            kernel_sizes=kernel_sizes,
            depths=depths,
            dims=dims,
            hidden_size=hidden_size,
            num_slices_list=num_slices_list,
            out_indices=out_indices,
            heads=heads,
        )
        self.out_channels = out_channels
        self.dims = dims
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        self.num_slices_list = num_slices_list
        self.num_tasks = num_tasks

        self.seg_decoder = Seg_Decoder(
            in_channels=hidden_size,
            dims=dims,
            out_chans=out_channels,
            kernel_sizes=kernel_sizes,
        )

        self.class_decoder = Class_Decoder(
            task=num_tasks,
            dims=dims,
            kernel_sizes=kernel_sizes,
        )

    def Upsample(self, x, size, align_corners=False):
        """
        Wrapper Around the Upsample Call
        """
        return nn.functional.interpolate(
            x, size=size, mode="trilinear", align_corners=align_corners
        )

    def forward(self, x):
        deep_feature, feature_out = self.encoder(x)

        CLS_out = self.class_decoder(feature_out)

        SEG_out = self.seg_decoder(deep_feature, feature_out)

        # return CLS_out, SEG_out
        return CLS_out, SEG_out


if __name__ == "__main__":
    device = "cuda:0"

    x = torch.randn(size=(2, 3, 128, 128, 64)).to(device)
    # test_x = torch.randn(size=(2, 64, 88, 88)).to(device)

    model = HSL_Net(in_channels=3, out_channels=3, class_channels=1).to(device)

    CLS_out, SEG_out = model(x)

    print(CLS_out.size())
    print(SEG_out.size())
