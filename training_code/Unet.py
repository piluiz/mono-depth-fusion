"""A small U-Net for supervised metric depth fusion."""

import torch
from torch import nn
import torch.nn.functional as F


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
    )


class UNet(nn.Module):
    """Fuse RGB, three depth maps and two sensor masks into one depth map."""

    def __init__(self, in_channels=8):
        super().__init__()
        self.encoder1 = conv_block(in_channels, 16)
        self.encoder2 = conv_block(16, 32)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(32, 64)
        self.decoder2 = conv_block(64 + 32, 32)
        self.decoder1 = conv_block(32 + 16, 16)
        self.output = nn.Sequential(nn.Conv2d(16, 1, kernel_size=1), nn.Softplus())

    def forward(self, x):
        skip1 = self.encoder1(x)
        skip2 = self.encoder2(self.pool(skip1))
        x = self.bottleneck(self.pool(skip2))
        x = F.interpolate(x, size=skip2.shape[-2:], mode="nearest")
        x = self.decoder2(torch.cat((x, skip2), dim=1))
        x = F.interpolate(x, size=skip1.shape[-2:], mode="nearest")
        x = self.decoder1(torch.cat((x, skip1), dim=1))
        return self.output(x)


class FusionUNet(nn.Module):
    """Predict a metric correction using separate features for each input source."""

    input_keys = ("rgb", "depth_anything", "lidar", "stereo", "lidar_valid", "stereo_valid")

    def __init__(self, deep_channels=64, bottleneck_channels=96):
        # Defaults match FusionUNet checkpoints saved before widths were recorded.
        super().__init__()
        # Each source gets four full-resolution feature channels before fusion.
        self.rgb_stem = nn.Conv2d(3, 4, 3, padding=1)
        self.mono_stem = nn.Conv2d(1, 4, 3, padding=1)
        self.lidar_stem = nn.Conv2d(2, 4, 3, padding=1)
        self.stereo_stem = nn.Conv2d(2, 4, 3, padding=1)
        self.encoder1 = conv_block(16, 16)
        self.encoder2 = conv_block(16, 32)
        self.encoder3 = conv_block(32, deep_channels)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(deep_channels, bottleneck_channels)
        self.decoder3 = conv_block(bottleneck_channels + deep_channels, deep_channels)
        self.decoder2 = conv_block(deep_channels + 32, 32)
        self.decoder1 = conv_block(32 + 16, 16)
        self.correction = nn.Conv2d(16, 1, 1)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def forward(self, x):
        rgb, mono, lidar, stereo, lidar_valid, stereo_valid = x.split((3, 1, 1, 1, 1, 1), dim=1)
        lidar_features = self.lidar_stem(torch.cat((lidar * lidar_valid, lidar_valid), dim=1))
        stereo_features = self.stereo_stem(torch.cat((stereo * stereo_valid, stereo_valid), dim=1))
        features = torch.cat((self.rgb_stem(rgb), self.mono_stem(mono),
                              lidar_features, stereo_features), dim=1)
        skip1 = self.encoder1(F.relu(features))
        skip2 = self.encoder2(self.pool(skip1))
        skip3 = self.encoder3(self.pool(skip2))
        x = self.bottleneck(self.pool(skip3))
        for decoder, skip in ((self.decoder3, skip3), (self.decoder2, skip2), (self.decoder1, skip1)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = decoder(torch.cat((x, skip), dim=1))

        baseline = torch.where(lidar_valid > 0, lidar, torch.where(stereo_valid > 0, stereo, mono))
        return (baseline + self.correction(x)).clamp_min(0)
