import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = [
    "ResNet",
    "resnet10",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnet200",
    "get_fine_tuning_parameters",
]


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)
    ).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block,
        layers,
        spatial_size,
        sample_count,
        in_classes=2,
        shortcut_type="B",
        num_classes=400,
    ):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            in_classes,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)
        last_count = int(math.ceil(sample_count / 16))
        last_size = int(math.ceil(spatial_size / 32))

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.avgpool = nn.AvgPool3d(
        #     (last_count, last_size, last_size), stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)

        # x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


##########################################################################################
##########################################################################################


def get_fine_tuning_parameters(model, ft_prefixes):

    assert isinstance(ft_prefixes, str)

    if ft_prefixes == "":
        print("WARNING: training full network because --ft_predixes=None")
        return model.parameters()

    print("#" * 60)
    print("Setting finetuning layer prefixes: {}".format(ft_prefixes))

    ft_prefixes = ft_prefixes.split(",")
    parameters = []
    param_names = []
    for param_name, param in model.named_parameters():
        for prefix in ft_prefixes:
            if prefix in param_name:
                print("  Finetuning parameter: {}".format(param_name))
                parameters.append({"params": param, "name": param_name})
                param_names.append(param_name)

    for param_name, param in model.named_parameters():
        if param_name not in param_names:
            # This sames a lot of GPU memory...
            print("disabling gradient for: {}".format(param_name))
            param.requires_grad = False

    return parameters


# def get_fine_tuning_parameters(model, ft_begin_index):
#
#     assert isinstance(ft_begin_index, int)
#     if ft_begin_index == 0:
#         print('WARNING: training full network because --finetune_begin_index=0')
#         return model.parameters()
#
#     for param_name, param in model.named_modules():
#         print(param_name)
#
#
#     ft_module_names = []
#     for i in range(ft_begin_index, 5):
#         ft_module_names.append('layer{}'.format(i))
#     ft_module_names.append('fc')
#
#     print('Modules to finetune: {}'.format(ft_module_names))
#
#     parameters = []
#     param_names_to_finetune = []
#     for k, v in model.named_parameters():
#         for ft_module in ft_module_names:
#             if ft_module in k:
#                 parameters.append({'params': v, 'name': k})
#                 param_names_to_finetune.append(k)
#                 break
#         else:
#             parameters.append({'params': v, 'lr': 0.0, 'name': k})
#             param_names_to_finetune.append(k)
#
#     # Disabling gradients for frozen weights (hacky...)
#     frozen_module_names = []
#     for i in range(0, ft_begin_index):
#         frozen_module_names.append('layer{}'.format(i))
#     for k, v in model.named_parameters():
#         for frozen_module in frozen_module_names:
#             if frozen_module in k:
#                 print('disabling grad for: {}'.format(k))
#                 v.requires_grad = False
#     model.module.conv1.requires_grad = False
#     model.module.bn1.requires_grad = False
#
#     return parameters

##########################################################################################
##########################################################################################


def resnet10(**kwargs):
    """Constructs a ResNet-18 model."""
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model."""
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model."""
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model."""
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model."""
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model."""
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model."""
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = resnet50(
        in_classes=2,
        num_classes=1,
        shortcut_type="B",
        spatial_size=64,
        sample_count=128,
    ).to(device)

    x = torch.randn(2, 2, 128, 128, 64).to(device)
    y = model(x)
    print(y.size())
