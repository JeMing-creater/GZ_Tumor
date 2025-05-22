import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


class MultiTaskSwinUNETR(nn.Module):
    def __init__(self, img_size=(128, 128, 64), in_channels=3, num_tasks=2, num_classes_per_task=1):
        super(MultiTaskSwinUNETR, self).__init__()
        # 初始化SwinUNETR作为共享的特征提取器
        self.feature_extractor = SwinUNETR(img_size=img_size, in_channels=in_channels, out_channels=num_classes_per_task)
        
        # 添加全局平均池化层
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # 为每个任务定义独立的分类头
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_classes_per_task, 64),  # 全连接层
                nn.ReLU(),
                nn.Linear(64, num_classes_per_task)  # 输出层
            ) for _ in range(num_tasks)
        ])
    
    def forward(self, x):
        # 提取特征
        features = self.feature_extractor(x)
        features = self.global_avg_pool(features)
        features = torch.flatten(features, start_dim=1)
        
        # 对每个任务应用独立的分类头，并收集结果
        task_outputs = [task_head(features) for task_head in self.task_heads]
        
        # 在channels维度上拼接所有任务的输出
        concatenated_output = torch.stack(task_outputs, dim=1)
        
        return concatenated_output  # 返回形状为(batch_size, channels, label)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskSwinUNETR()  # 根据实际需求调整参数
    model.to(device)
    
    x = torch.randn(size=(2, 3, 128, 128, 64)).to(device)
    
    print(model(x).shape)