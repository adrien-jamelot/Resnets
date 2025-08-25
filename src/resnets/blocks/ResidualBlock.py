from torch.nn import Conv2d, ReLU, BatchNorm2d
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv_1 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=1,
        )
        self.conv_2 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=1,
        )
        self.batchNorm_1 = BatchNorm2d(out_channels)
        self.batchNorm_2 = BatchNorm2d(out_channels)
        self.convBlock1 = nn.Sequential(self.conv_1, self.batchNorm_1, ReLU())
        self.convBlock2 = nn.Sequential(self.conv_2, self.batchNorm_2, ReLU())

    def forward(self, x):
        out1 = self.convBlock1(x)
        out2 = self.convBlock2(out1)
        return out2 + x


class ResidualBlockWithDownsampling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.downsampleResidual = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=1,
        )
        self.conv_1 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=1,
        )
        self.conv_2 = Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
        )
        self.batchNorm_1 = BatchNorm2d(out_channels)
        self.batchNorm_2 = BatchNorm2d(out_channels)
        self.convBlock1 = nn.Sequential(self.conv_1, self.batchNorm_1, ReLU())
        self.convBlock2 = nn.Sequential(self.conv_2, self.batchNorm_2, ReLU())

    def forward(self, x):
        out1 = self.convBlock1(x)
        out2 = self.convBlock2(out1)
        return out2 + self.downsampleResidual(x)
