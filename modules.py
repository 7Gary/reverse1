import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Simple squeeze-excitation block used by PAM."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class PeriodicityAwareModule(nn.Module):
    """Extracts periodic texture cues with lightweight spectral priors."""

    def __init__(self, channels: int, reduction: int = 16, directional_kernel: int = 3):
        super().__init__()
        self.directional = nn.Conv2d(
            channels,
            channels,
            kernel_size=directional_kernel,
            padding=directional_kernel // 2,
            groups=channels,
            bias=False,
        )
        self.freq_project = nn.Conv2d(1, channels, kernel_size=1)
        self.ca = ChannelAttention(channels, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frequency magnitude (spectral prior)
        freq = torch.fft.fft2(x, norm="ortho")
        freq = torch.abs(freq)
        freq = torch.mean(freq, dim=1, keepdim=True)
        freq = F.avg_pool2d(freq, kernel_size=3, stride=1, padding=1)
        freq = self.freq_project(freq)

        # Directional consistency
        direction = self.directional(x)

        fused = x + direction + freq
        fused = self.ca(fused)
        return fused


class LocalTextureSelfAttention(nn.Module):
    """Models local-and-neighbor texture continuity for anomaly cues."""

    def __init__(self, channels: int, dilation: int = 2):
        super().__init__()
        self.local = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=1, bias=False)
        self.neighbor = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=1,
            bias=False,
        )
        self.fuse = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_feat = self.local(x)
        neighbor_feat = self.neighbor(x)
        weight = torch.sigmoid(self.fuse(torch.cat([local_feat, neighbor_feat], dim=1)))
        refined = local_feat + weight * (neighbor_feat - local_feat)
        refined = self.act(self.norm(refined))
        return refined + x