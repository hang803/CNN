import torch
import torch.nn as nn

layer = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=0)  # 输入为单通道灰度图像，用3个3*3的卷积核卷积，步长为1，未加padding
x = torch.rand(1, 1, 28, 28)
y = layer.forward(x)
print(y.shape)
layer1 = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)  # 增加1padding，卷积后的图片尺寸不变
y1 = layer1.forward(x)
print(y1.shape)
layer2 = nn.Conv2d(1, 3, kernel_size=3, stride=2, padding=1)  # 步长2，尺寸x，y方向各变为一半
y2 = layer2.forward(x)
print(y2.shape)
print(layer.weight)    # 查看layer的权重
print(layer.weight.shape)    # torch.Size([3, 1, 3, 3])
print(layer.bias.shape)    # torch.Size([3])
