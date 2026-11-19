import torch
import torch.nn as nn
import torch.nn.functional as F


class TraversabilityDecoder(nn.Module):
    def __init__(self, in_channels=(40, 112, 320), mid_channels=64):
        super().__init__()

        self.lateral = nn.ModuleList([
            nn.Conv2d(c, mid_channels, kernel_size=1) for c in in_channels
        ])

        self.deconv1 = nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=4, stride=2, padding=1)

        self.smooth1 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, mid_channels),
            nn.ReLU(),
        )
        self.smooth2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, mid_channels),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, s8, s16, s32):
        x1 = self.lateral[0](s8)
        x2 = self.lateral[1](s16)
        x3 = self.lateral[2](s32)

        up = self.deconv1(x3)
        if up.shape[-2:] != x2.shape[-2:]:
            up = F.interpolate(up, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        x2 = self.smooth1(x2 + up)

        up = self.deconv2(x2)
        if up.shape[-2:] != x1.shape[-2:]:
            up = F.interpolate(up, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        x1 = self.smooth2(x1 + up)

        out = self.head(x1)
        out = F.interpolate(out, size=(64, 64), mode="bilinear", align_corners=False)
        out = self.refine(out)
        return self.sigmoid(out)


class TraversabilityTokenizer(nn.Module):
    def __init__(self, grid_size=2, embed_dim=256, in_channels=1):
        super().__init__()

        self.grid_size = grid_size
        self.embed_dim = embed_dim

        self.downsample1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )

        self.downsample2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )

        self.downsample3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(grid_size)
        self.proj = nn.Conv2d(128, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.flatten = nn.Flatten(start_dim=2)

    def forward(self, x):
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)

        x = self.avg_pool(x)
        x = self.proj(x)
        x = self.flatten(x)
        x = x.transpose(1, 2)
        x = self.norm(x)

        return x



def distil_loss(pred, target, mode):
    if mode == "dino":
        target = target.to(device=pred.device, dtype=pred.dtype)
        return torch.mean(torch.abs(pred - target))
    else: 
        raise ValueError


def get_teacher_map(mode, teacher_batch, actions=None):
    if mode == "dino":
        return teacher_batch.to(dtype=torch.float32)
    if mode == "path":
        raise NotImplementedError("path mode needs rasterize_path, which is not implemented")
    raise ValueError(f"Unknown teacher mode: {mode}")
