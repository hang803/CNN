import torch
import torch.nn as nn
layer=nn.Conv2d(1,3,kernel_size=3,stride=1,padding=1)
x=torch.rand(1,1,28,28)
y=layer(x)
print(y.shape)
layer1=nn.MaxPool2d(kernel_size=2,stride=2)
y1=layer1(y)
print(y1.shape)
layer2=nn.Upsample(scale_factor=2,mode='nearest')#upsample增加尺寸，它是复制数据
y2=layer2(y1)
print(y2.shape)