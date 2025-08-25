import torch
from torch import Tensor
from torch.nn import AvgPool2d, Conv2d, ReLU, BatchNorm1d, AdaptiveAvgPool2d
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        self.conv_1 = Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size
        )
        self.conv_2 = Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size
        )
        self.batchNorm_1 = BatchNorm1d(3)
        self.batchNorm_2 = BatchNorm1d(3)
        self.convBlock1 = nn.Sequential(self.conv_1, self.batchNorm_1, ReLU())
        self.convBlock2 = nn.Sequential(self.conv_2, self.batchNorm_2, ReLU())

    def forward(self, x):
        out1 = self.convBlock1(x)
        out2 = self.convBlock2(out1)
        return out2 + x


class ResidualBlockWithDownsampling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        self.downsampleResidual = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2,
        )
        self.conv_1 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2,
        )
        self.conv_2 = Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
        )
        self.batchNorm_1 = BatchNorm1d(3)
        self.batchNorm_2 = BatchNorm1d(3)
        self.convBlock1 = nn.Sequential(self.conv_1, self.batchNorm_1, ReLU())
        self.convBlock2 = nn.Sequential(self.conv_2, self.batchNorm_2, ReLU())

    def forward(self, x):
        out1 = self.convBlock1(x)
        out2 = self.convBlock2(out1)
        return out2 + self.downsampleResidual(x)


class ResNet18(nn.Module):
    def __init__(self):
        self.conv1_x = Conv2d(3, 64, 7, 2)
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
        self.fc = nn.Linear(512, 1000)
        self.forwardSeq = nn.Sequential(
            self.conv1_x,
            self.conv2_x,
            self.conv3_x,
            self.conv4_x,
            self.conv5_x,
            self.global_average_pooling,
            self.fc,
        )

    def forward(self, x: Tensor):
        return self.forwardSeq(x)
