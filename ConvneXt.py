import  torch
import  torch.nn as nn
import random
import math
import  torch.optim as optim
from    torchvision import datasets, transforms  #torch数据预处理
from torchsummary import summary

class residual(nn.Module):
     def __init__(self,sort,in_channel,out_channel,strides=1,change_channel=True,stochastic_depth=0):
          super(residual, self).__init__()
          self.conv1=nn.Conv2d(in_channel,out_channel,kernel_size=7,stride=strides,padding=3,groups=in_channel)
          self.conv2=nn.Conv2d(out_channel,4*out_channel,kernel_size=1)
          self.conv3=nn.Conv2d(4*out_channel,out_channel,kernel_size=1)
          if change_channel:
               self.conv=nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=strides)
          self.has_conv1=change_channel
          ratio=2**(sort-1)
          self.ln=nn.LayerNorm((56//ratio,56//ratio))
          self.gelu=nn.ReLU()
          self.stochastic_depth=stochastic_depth
          self.floor=math.log(out_channel/96,2)+1
          self.rate=1-self.floor/4*(1-0.5)

     def forward(self, x):
          p=random.random()
          if self.has_conv1:           #change channel:can not use stochastic
               y1 = self.ln(self.conv1(x))
               y2 = self.gelu(self.conv2(y1))
               y3 = self.conv3(y2)
               x=self.conv(x)
               y=y3+x
          elif self.stochastic_depth==0: #not use
               y1 = self.ln(self.conv1(x))
               y2 = self.gelu(self.conv2(y1))
               y3 = self.conv3(y2)
               y=y3+x
          elif self.stochastic_depth==1: #p=define value
               if random.random()>0.5:
                    y=x
               else:
                    y1 = self.ln(self.conv1(x))
                    y2 = self.gelu(self.conv2(y1))
                    y3 = self.conv3(y2)
                    y=y3+x
          elif self.stochastic_depth==2: #p is higher with floor deeper
               if random.random()>self.rate:
                    y=x
               else:
                    y1 = self.ln(self.conv1(x))
                    y2 = self.gelu(self.conv2(y1))
                    y3 = self.conv3(y2)
                    y=y3+x

          return y
def conv_blk(sort,in_channel,out_channel,num_blocks,first_blk=False,stochastic_mode=1):
     blk=[]
     for i in range(num_blocks):
          if i==0 and not first_blk:
               res=residual(sort,in_channel,out_channel,strides=2,stochastic_depth=stochastic_mode)
               blk.append(res)
          else:
               res=residual(sort,out_channel,out_channel,change_channel=False,stochastic_depth=stochastic_mode)
               blk.append(res)
     return blk
r2=conv_blk(1,96,96,num_blocks=3,first_blk=True)
r3=conv_blk(2,96,192,num_blocks=3,)
r4=conv_blk(3,192,384,num_blocks=9,)
r5=conv_blk(4,384,768,num_blocks=3,)

b1=nn.Conv2d(1,96,kernel_size=4,stride=4)
b2=nn.Sequential(*r2)
b3=nn.Sequential(*r3)
b4=nn.Sequential(*r4)
b5=nn.Sequential(*r5)

ConvneXt_T=nn.Sequential(b1,b2,b3,b4,b5,
                         nn.AdaptiveAvgPool2d((1,1)),
                         nn.Flatten(),
                         nn.Linear(768,100))

summary(ConvneXt_T,input_size=[(1,224,224)],batch_size=32,device='cpu')

def trans(img_size,randomhorizontalflip,randomverticalflip,randomresizecrop,randomrotation):
     trans=[]
     if randomhorizontalflip:
          trans.append(transforms.RandomHorizontalFlip())
     if randomverticalflip:
          trans.append(transforms.RandomVerticalFlip())
     if randomresizecrop:
          trans.append(transforms.RandomResizedCrop(size=randomresizecrop[:1],
                                                    scale=randomresizecrop[1:3],
                                                    ratio=randomresizecrop[-2:]))
     if randomrotation:
          trans.append(transforms.RandomRotation(randomrotation))
     transform=transforms.Compose([transforms.Resize(img_size),*trans,transforms.ToTensor()]
                                  )
     return transform
a=trans(img_size=[224,224],randomhorizontalflip=True,randomverticalflip=True,
        randomresizecrop=[224,0.1,1,0.5,2],randomrotation=(-45,45))
print(a)

