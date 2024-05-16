# NOTICE: This is kornia's implementation of UNet's blocks see
# https://github.com/kornia/kornia/blob/a041d9255f459a0034c75b20fb9ab524d7ae3983/kornia/feature/disk/_unets/blocks.py

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class TrivialUpsample(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


class TrivialDownsample(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, 2)


class Conv(nn.Sequential):
    def __init__(self, in_: int, out_: int, size: int, skip_norm_and_gate: bool = False) -> None:
        norm: nn.Module
        nonl: nn.Module

        if skip_norm_and_gate:
            norm = nn.Sequential()
            nonl = nn.Sequential()
        else:
            norm = nn.InstanceNorm2d(in_)
            nonl = nn.PReLU(in_)

        dropout = nn.Sequential()
        conv = nn.Conv2d(in_, out_, size, padding="same", bias=True)

        super().__init__(norm, nonl, dropout, conv)


class ThinUnetDownBlock(nn.Sequential):
    def __init__(self, in_: int, out_: int, size: int = 5, is_first: bool = False, setup: Any = None) -> None:
        self.in_ = in_
        self.out_ = out_

        downsample: nn.Module
        if is_first:
            downsample = nn.Sequential()
            conv = Conv(in_, out_, size, skip_norm_and_gate=True)
        else:
            downsample = TrivialDownsample()
            conv = Conv(in_, out_, size)

        super().__init__(downsample, conv)


class ThinUnetUpBlock(nn.Module):
    def __init__(self, bottom_: int, horizontal_: int, out_: int, size: int = 5, setup: Any = None) -> None:
        super().__init__()

        self.bottom_ = bottom_
        self.horizontal_ = horizontal_
        self.cat_ = bottom_ + horizontal_
        self.out_ = out_

        self.upsample = TrivialUpsample()
        self.conv = Conv(self.cat_, self.out_, size)

    def forward(self, bot: torch.Tensor, hor: torch.Tensor) -> torch.Tensor:
        bot_big = self.upsample(bot)
        combined = torch.cat([bot_big, hor], dim=1)

        return self.conv(combined)
