import torch
import torch.nn as nn
x=torch.rand(100,16,784)#（0,1）均匀分布，均值为0.5，方差为1
layer=nn.BatchNorm1d(16)
y=layer(x)
print(layer.running_mean)
print(layer.running_var)

x1=torch.rand(1,16,7,7)
layer1=nn.BatchNorm2d(16)
y1=layer1(x1)
print(y1.shape)
print(layer1.weight)  #均值
print(layer1.bias)  #方差
print(vars(layer1))  #train代表是否是训练的，affine代表是否更新数据

'''
用于卷积层参数设置为channels
用于线性层参数设置为输出的大小
'''
