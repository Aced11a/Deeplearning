import torch
import torch.nn as nn

class SPPF(nn.Module):
    """
    YOLOv5 SPPF (Spatial Pyramid Pooling — Fast)
    输入:  (B, C, H, W)
    输出:  (B, C_out, H, W)

    默认 k=5 对应 (5×5) 最大池化，重复 3 次拼接
    """
    def __init__(self, c1, c2, k=5):
        """
        c1: 输入通道
        c2: 输出通道
        k : 最大池化核大小
        """
        super().__init__()
        self.cv1 = nn.Conv2d(c1, c1//2, kernel_size=1, stride=1)
        self.cv2 = nn.Conv2d(c1//2 * 4, c2, kernel_size=1, stride=1)

        # 取 k=5 的池化层，stride=1,padding=2
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)

        # 四个特征图 concat: x, y1, y2, y3
        return self.cv2(torch.cat((x, y1, y2, y3), dim=1))


# 测试
if __name__ == "__main__":
    x = torch.randn(1, 64, 20, 20)
    sppf = SPPF(64, 128)
    y = sppf(x)
    print("Output:", y.shape)
