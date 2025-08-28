from torch import Tensor
from torch.nn import Conv2d, AdaptiveAvgPool2d, MaxPool2d
from torch import nn
from resnets.blocks.ResidualBlock import (
    ResidualBlockWithDownsampling,
    ResidualBlock,
    LMBlock,
)
import torch


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_x = Conv2d(3, 64, 7, 2)
        self.max_pooling1 = MaxPool2d(3, 2, 1)
        self.conv2_x = nn.Sequential(ResidualBlock(64, 64, 3), ResidualBlock(64, 64, 3))
        self.conv3_x = nn.Sequential(
            ResidualBlockWithDownsampling(64, 128, 3), ResidualBlock(128, 128, 3)
        )
        self.conv4_x = nn.Sequential(
            ResidualBlockWithDownsampling(128, 256, 3), ResidualBlock(256, 256, 3)
        )
        self.conv5_x = nn.Sequential(
            ResidualBlockWithDownsampling(256, 512, 3), ResidualBlock(512, 512, 3)
        )
        self.global_average_pooling = AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)

    def forward(self, x: Tensor):
        conv1 = self.conv1_x(x)
        pool1 = self.max_pooling1(conv1)
        conv2 = self.conv2_x(pool1)
        conv3 = self.conv3_x(conv2)
        conv4 = self.conv4_x(conv3)
        conv5 = self.conv5_x(conv4)
        gap = self.global_average_pooling(conv5).flatten(1)
        fc = self.fc(gap)
        return fc


class ResNetMini(nn.Module):
    def __init__(self, scale=16):
        super().__init__()
        self.conv1_x = Conv2d(3, scale, 7, 2)
        self.max_pooling1 = MaxPool2d(3, 2, 1)
        self.conv2_x = nn.Sequential(ResidualBlock(scale, scale, 3))
        self.global_average_pooling = AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(scale, 10)

    def forward(self, x: Tensor):
        conv1 = self.conv1_x(x)
        pool1 = self.max_pooling1(conv1)
        conv2 = self.conv2_x(pool1)
        gap = self.global_average_pooling(conv2).flatten(1)
        fc = self.fc(gap)
        return fc


class ResNetMiniDeep(nn.Module):
    def __init__(self, scale=1):
        super().__init__()
        self.conv1_x = Conv2d(3, scale, 3, 2)
        self.max_pooling1 = MaxPool2d(3, 2, 1)
        self.conv2_x = nn.ModuleList(
            [ResidualBlock(scale, scale, 3) for _ in range(128)]
        )
        self.global_average_pooling = AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(scale, 10)

    def forward(self, x: Tensor):
        conv1 = self.conv1_x(x)
        pool1 = self.max_pooling1(conv1)
        conv2 = pool1
        for block in self.conv2_x:
            conv2 = block(conv2)
        gap = self.global_average_pooling(conv2).flatten(1)
        fc = self.fc(gap)
        return fc


class LMResNetMiniDeep(nn.Module):
    def __init__(self, scale=16):
        super().__init__()
        self.conv1_x = Conv2d(3, scale, 3, 2)
        self.max_pooling1 = MaxPool2d(3, 2, 1)
        self.initLM = ResidualBlock(scale, scale, 3)
        self.conv2_x = nn.ModuleList([LMBlock(scale, scale, 3) for _ in range(2)])
        self.downSample = ResidualBlockWithDownsampling(scale, scale * 2, 3)
        self.conv3_x = nn.ModuleList(
            [LMBlock(scale * 2, scale * 2, 3) for _ in range(2)]
        )
        self.global_average_pooling = AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(scale, 10)

    def forward(self, x: Tensor):
        conv1 = self.conv1_x(x)
        pool1 = self.max_pooling1(conv1)
        conv2 = torch.concat(
            [pool1.unsqueeze(-1), self.initLM(pool1).unsqueeze(-1)], -1
        )
        for block in self.conv2_x:
            conv2 = block(conv2)
        conv2 = conv2[..., 1]
        gap = self.global_average_pooling(conv2).flatten(1)
        fc = self.fc(gap)
        return fc
