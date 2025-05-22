import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding3D(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=(16, 16, 16), img_size=(128, 128, 64)):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Compute number of patches
        self.num_patches = (img_size[0] // patch_size[0]) * \
                           (img_size[1] // patch_size[1]) * \
                           (img_size[2] // patch_size[2])

        # Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x):
        x = self.proj(x)  # shape: [B, embed_dim, D', H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches+1, embed_dim]
        x = x + self.pos_embed
        return x


class ViT(nn.Module):
    def __init__(self, in_channels=1, img_size=(128, 128, 64),
                 patch_size=(16, 16, 16), embed_dim=768, depth=12, num_heads=12, mlp_dim=3072):
        super().__init__()
        self.patch_embed = PatchEmbedding3D(in_channels, embed_dim, patch_size, img_size)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=mlp_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            # nn.Sigmoid()
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出层
        )

    def forward(self, x):
        x = self.patch_embed(x)  # [B, N_patches+1, embed_dim]
        x = self.transformer(x)
        cls_token = x[:, 0]  # [B, embed_dim]
        out = self.classifier(cls_token)
        return out


# 示例使用
model = ViT(in_channels=1, img_size=(128, 128, 64))
x = torch.randn(2, 1, 128, 128, 64)  # (B, C, D, H, W)
output = model(x)
print(output.shape)  # 应为 (2, 1)
