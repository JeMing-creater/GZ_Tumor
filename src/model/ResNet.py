import torch
import torch.nn as nn
import torchvision

# 基于 Torchvision 的 ResNet 模板构建 3D ResNet
from torchvision.models.video import r3d_18  # 可选替代 ResNet-50 的轻量版本

class ResNet3DClassifier(nn.Module):
    def __init__(self, in_channels=1, pretrained=False):
        super(ResNet3DClassifier, self).__init__()
        self.model = torchvision.models.video.r3d_18(pretrained=pretrained)
        
        # 替换第一层卷积以适配输入通道数
        self.model.stem[0] = nn.Conv3d(
            in_channels, 64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False
        )

        # 替换最后的全连接层，输出为一个值（用于二分类）
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出层
        )

    def forward(self, x):
        return self.model(x)

# 示例输入
# 假设 class_num = 1
model = ResNet3DClassifier(in_channels=1)
x = torch.randn(2, 1, 128, 128, 64)  # (batch, channel, D, H, W)
output = model(x)
print(output.shape)  # 应为 (2, 1)
