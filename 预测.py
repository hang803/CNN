import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F

from PIL import Image

from torchvision import transforms
class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1,6,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.Conv2d(6,16,kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.Flatten(),
            nn.Linear(16*5*5,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10),

        )

    def forward(self, x):
        x = self.model(x)

        return x
device=torch.device('cuda:0')

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] #这个顺序很重要，要和训练时候的类名顺序一致

m=torch.load('model.pt')


img = cv2.imread('0.png')
img = cv2.resize(img, (28, 28))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
img=~(img)
cv2.imshow('gg',img)

img_ = torch.from_numpy(img).float().view(-1,1,28,28)/255


img_ = img_.to(device)
outputs=m(img_)

print(outputs)
print(F.softmax(outputs,dim=1))
print(F.cross_entropy(F.softmax(outputs,dim=1),torch.tensor([6]).cuda()))
_, indices = torch.max(outputs,1)
print(indices)
percentage = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
perc = percentage[int(indices)].item()
result = class_names[indices]
print('predicted:', result)
print('perc',perc)
cv2.waitKey(0)