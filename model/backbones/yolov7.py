from torch import nn
from ..layers import BaseConv, ConcatBlock


class Yolov7TinyBackbone(nn.Module):
    def __init__(self, act="ReLU"):
        super().__init__()

        self.stem = nn.Sequential(
            BaseConv(3, 32, 3, stride=2, act=act),
            BaseConv(32, 64, 3, stride=2, act=act),
        )

        self.concatblock0 = ConcatBlock(in_channels=64, out_channels=32, act=act)
        self.concatblock1 = ConcatBlock(in_channels=64, out_channels=64, act=act)
        self.concatblock2 = ConcatBlock(in_channels=128, out_channels=128, act=act)
        self.concatblock3 = ConcatBlock(in_channels=256, out_channels=256, act=act)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.stem(x)

        x0 = self.concatblock0(x)
        x = self.maxpool(x0)
        x1 = self.concatblock1(x)
        x = self.maxpool(x1)
        x2 = self.concatblock2(x)
        x = self.maxpool(x2)
        x3 = self.concatblock3(x)

        return x0, x1, x2, x3


def yolov7_tiny(**kwargs):
    return Yolov7TinyBackbone(act=kwargs["act"])
