import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
class resdidual(nn.Module):
    def __init__(self,in_channels,num_channels,use_1x1convd=False,strides=1):
        super(resdidual, self).__init__()
        self.conv1=nn.Conv2d(in_channels,num_channels,kernel_size=3,padding=1,stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1convd:
            self.conv3=nn.Conv2d(in_channels,num_channels,kernel_size=1,stride=strides)
        else:
            self.conv3=None
        self.bn1=nn.BatchNorm2d(num_channels)
        self.bn2=nn.BatchNorm2d(num_channels)
        self.relu=nn.ReLU(inplace=True)
    def forward(self, x):
        y=F.relu(self.bn1(self.conv1(x)))
        y=self.bn2(self.conv2(y))
        if self.conv3:
            x=self.conv3(x)
        y+=x
        return self.relu(y)
b1=nn.Sequential(
    nn.Conv2d(1,16,kernel_size=7,stride=2,padding=3),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
)
def res_blk(in_channels,num_channels,num_residuals,first_blk=False):
    blk=[]
    for i in range(num_residuals):
        if i==0 and not first_blk:
            blk.append(resdidual(in_channels,num_channels,use_1x1convd=True,strides=2))
        else:
            blk.append(resdidual(num_channels,num_channels))
    return blk
b2=nn.Sequential(*res_blk(16,16,2,first_blk=True))
b3=nn.Sequential(*res_blk(16,32,2))
b4=nn.Sequential(*res_blk(32,64,2))
b5=nn.Sequential(*res_blk(64,128,2))
device = torch.device('cuda:0')
net =nn.Sequential(b1,b2,b3,b4,b5,
                   nn.AdaptiveAvgPool2d((1,1)),
                   nn.Flatten(),
                   nn.Linear(128,10)
                   ).to(device)
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
net.load_state_dict(torch.load("ResNet.pkl"))
net.eval()

img = cv2.imread('0.png')
img = cv2.resize(img, (28, 28))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
img=~(img)
cv2.imshow('gg',img)
img_ = torch.from_numpy(img).float().view(-1,1,28,28)/255


img_ = img_.to(device)
outputs=net(img_)


_, indices = torch.max(outputs,1)
percentage = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
perc = percentage[int(indices)].item()
result = class_names[indices]
print('predicted:', result)
print('perc',perc)
cv2.waitKey(0)

