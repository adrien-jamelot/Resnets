from torch import Tensor
from torch.nn import Conv2d, AdaptiveAvgPool2d, MaxPool2d
from torch import nn
from resnets.blocks.ResidualBlock import ResidualBlockWithDownsampling, ResidualBlock


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
