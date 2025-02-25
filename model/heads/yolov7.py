from torch import nn
from ..layers import BaseConv


class Yolov7Head(nn.Module):
    def __init__(self, in_channels=[64, 128, 256], act="ReLU"):
        super().__init__()

        self.conv11 = BaseConv(in_channels[0], 128, 3, stride=1, act=act)
        self.outlayer1 = nn.Conv2d(128, 5, kernel_size=1)

        self.conv12 = BaseConv(in_channels[1], 256, 3, stride=1, act=act)
        self.outlayer2 = nn.Conv2d(256, 5, kernel_size=1)

        self.conv13 = BaseConv(in_channels[2], 512, 3, stride=1, act=act)
        self.outlayer3 = nn.Conv2d(512, 5, kernel_size=1)

    def forward(self, x):
        out1, out2, out3 = x

        out1 = self.conv11(out1)
        out1 = self.outlayer1(out1)

        out2 = self.conv12(out2)
        out2 = self.outlayer2(out2)

        out3 = self.conv13(out3)
        out3 = self.outlayer3(out3)

        return out1, out2, out3
