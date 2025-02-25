import torch
from torch import nn
from ..layers import BaseConv


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        expansion=0.5,
        act="ReLU",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(hidden_channels, out_channels, 3, stride=1, act=act)

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return y


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        expansion=0.5,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(hidden_channels, hidden_channels, 1.0, act=act) for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class YoloXNeck(nn.Module):
    def __init__(
        self,
        num_channels_in=[128, 256, 512],
        num_channels_out=[128, 256, 512],
        act="ReLU",
    ):
        super().__init__()
        self.num_channels_in = num_channels_in
        self.num_channels_out = num_channels_out

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            self.num_channels_in[2], self.num_channels_in[1], 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            in_channels=2 * self.num_channels_in[1],
            out_channels=self.num_channels_out[1],
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            self.num_channels_out[1], self.num_channels_in[0], 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            in_channels=2 * self.num_channels_in[0],
            out_channels=self.num_channels_out[0],
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = BaseConv(
            self.num_channels_out[0], self.num_channels_in[0], 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            in_channels=2 * self.num_channels_in[0],
            out_channels=self.num_channels_out[1],
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = BaseConv(
            self.num_channels_out[1], self.num_channels_in[1], 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            in_channels=2 * self.num_channels_in[1],
            out_channels=self.num_channels_out[2],
            act=act,
        )

    def forward(self, blocks):
        x2, x1, x0 = blocks

        fpn_out0 = self.lateral_conv0(x0)
        f_out0 = self.upsample(fpn_out0)
        f_out0 = torch.cat([f_out0, x1], dim=1)
        f_out0 = self.C3_p4(f_out0)

        fpn_out1 = self.reduce_conv1(f_out0)
        f_out1 = self.upsample(fpn_out1)
        f_out1 = torch.cat([f_out1, x2], dim=1)
        pan_out2 = self.C3_p3(f_out1)

        p_out1 = self.bu_conv2(pan_out2)
        p_out1 = torch.cat([p_out1, fpn_out1], dim=1)
        pan_out1 = self.C3_n3(p_out1)

        p_out0 = self.bu_conv1(pan_out1)
        p_out0 = torch.cat([p_out0, fpn_out0], dim=1)
        pan_out0 = self.C3_n4(p_out0)

        return pan_out2, pan_out1, pan_out0
