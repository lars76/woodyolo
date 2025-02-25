import torch
from torch import nn
from ..layers import BaseConv, ConcatBlock


class Yolov7Neck(nn.Module):
    def __init__(self, num_channels_in, num_channels_out, act="ReLU"):
        super().__init__()
        self.num_channels_in = num_channels_in
        self.num_channels_out = num_channels_out

        self.conv1 = BaseConv(self.num_channels_in[2], 256, 1, stride=1, act=act)
        self.conv2 = BaseConv(self.num_channels_in[2], 256, 1, stride=1, act=act)

        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)

        self.conv3 = BaseConv(1024, 256, 1, stride=1, act=act)
        self.conv4 = BaseConv(512, 256, 1, stride=1, act=act)
        self.conv5 = BaseConv(256, 128, 1, stride=1, act=act)

        self.upsampling = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv6 = BaseConv(self.num_channels_in[1], 128, 1, stride=1, act=act)

        self.concatblock4 = ConcatBlock(in_channels=256, out_channels=64, act=act)
        self.conv7 = BaseConv(128, 64, 1, stride=1, act=act)

        self.conv8 = BaseConv(self.num_channels_in[0], 64, 1, stride=1, act=act)

        self.concatblock5 = ConcatBlock(
            in_channels=128, out_channels=self.num_channels_out[0] // 2, act=act
        )

        self.conv9 = BaseConv(self.num_channels_out[0], 128, 3, stride=2, act=act)

        self.concatblock6 = ConcatBlock(
            in_channels=256, out_channels=self.num_channels_out[1] // 2, act=act
        )

        self.conv10 = BaseConv(self.num_channels_out[1], 256, 3, stride=2, act=act)

        self.concatblock7 = ConcatBlock(
            in_channels=512, out_channels=self.num_channels_out[2] // 2, act=act
        )

    def forward(self, x):
        f1, f2, f3 = x
        # print(f1.shape, f2.shape, f3.shape)

        f1 = self.conv8(f1)
        f2 = self.conv6(f2)
        x0 = self.conv1(f3)

        x1 = self.conv2(f3)
        m1 = self.maxpool1(x1)
        m2 = self.maxpool2(x1)
        m3 = self.maxpool3(x1)
        concat = torch.cat((m1, m2, m3, x1), dim=1)
        concat = self.conv3(concat)

        concat2 = torch.cat((concat, x0), dim=1)
        concat2_ = self.conv4(concat2)

        concat2 = self.conv5(concat2_)

        concat2 = self.upsampling(concat2)

        concat3 = torch.cat((concat2, f2), dim=1)
        concat3 = self.concatblock4(concat3)
        k = concat3
        concat3 = self.conv7(concat3)

        concat3 = self.upsampling(concat3)

        concat4 = torch.cat((concat3, f1), dim=1)
        concat4 = self.concatblock5(concat4)
        out1 = concat4
        concat4 = self.conv9(concat4)

        concat5 = torch.cat((concat4, k), dim=1)
        concat5 = self.concatblock6(concat5)
        out2 = concat5
        concat5 = self.conv10(concat5)

        concat5 = torch.cat((concat2_, concat5), dim=1)
        concat5 = self.concatblock7(concat5)
        out3 = concat5

        return out1, out2, out3
