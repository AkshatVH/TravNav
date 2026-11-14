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