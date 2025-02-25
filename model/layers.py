from torch import nn
import torch


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, act="ReLU"):
        super().__init__()
        if stride != 1 and kernel_size > 1:
            padding = 1
        else:
            padding = "same"
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = getattr(torch.nn, act)(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ConcatBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act="ReLU"):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = BaseConv(self.in_channels, self.out_channels, 1, 1, act)
        self.conv2 = BaseConv(self.in_channels, self.out_channels, 1, 1, act)
        self.conv3 = BaseConv(self.out_channels, self.out_channels, 3, 1, act)
        self.conv4 = BaseConv(self.out_channels, self.out_channels, 3, 1, act)
        self.conv5 = BaseConv(self.out_channels * 4, self.out_channels * 2, 1, 1, act)

    def forward(self, x):
        x0 = self.conv1(x)

        x1 = self.conv2(x)
        x2 = self.conv3(x1)
        x3 = self.conv4(x2)

        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.conv5(x)

        return x
