import torch
import torch.nn as nn

class Focus(nn.Module):
    """
    YOLOv5 Focus module:
    把输入特征图按奇偶行列分成4份，在通道维度上拼接，再卷积。
    输入: (B, C, H, W)
    输出: (B, 4*C_out_channels, H/2, W/2) 经过 conv 后变成 (B, out_channels, H/2, W/2)
    """
    def __init__(self, in_channels, out_channels, k=1, s=1, p=None, groups=1, activation=True):
        super(Focus, self).__init__()
        # YOLOv5 里默认 padding = k//2（如果不给的话）
        if p is None:
            p = k // 2

        self.conv = nn.Conv2d(
            in_channels * 4,  # 因为拼了4块
            out_channels,
            kernel_size=k,
            stride=s,
            padding=p,
            groups=groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if activation else nn.Identity()

    def forward(self, x):
        # x: (B, C, H, W)
        # 切成4块: [::2,::2], [1::2,::2], [::2,1::2], [1::2,1::2]
        # 这样 H,W 变成一半，通道×4
        x1 = x[..., ::2, ::2]   # 偶行 偶列
        x2 = x[..., 1::2, ::2]  # 奇行 偶列
        x3 = x[..., ::2, 1::2]  # 偶行 奇列
        x4 = x[..., 1::2, 1::2] # 奇行 奇列

        x = torch.cat((x1, x2, x3, x4), dim=1)  # 在 channel 维度拼接
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# 简单测试
if __name__ == "__main__":
    x = torch.randn(1, 3, 640, 640)  # batch=1, C=3, H=W=640
    m = Focus(3, 32, k=3)            # in_channels=3, out_channels=32
    y = m(x)
    print(y.shape)  # 预期: (1, 32, 320, 320)

