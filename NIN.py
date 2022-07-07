import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PIL import Image
from torchvision import datasets, transforms  # torch数据预处理

from visdom import Visdom

batch_size = 100
learning_rate = 0.1
epochs = 10



train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([  # compose指将多个操作整合
                       transforms.Resize([224,224]),
                       transforms.ToTensor(),  # 将其转化为数据范围在0到1的张量
                       # transforms.Normalize((0.1307,), (0.3081,))
                   ])),  # 前面一个指令意为指定训练集
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True)  # batch_size表示批处理数据量，shuffle表示每次操作前打乱数据


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=0), nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=1), nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.ReLU(),
            nn.Conv2d(256,256, kernel_size=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Dropout(0.5),
            nn.Conv2d(384, 10, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(10, 10, kernel_size=1), nn.ReLU(),
            nn.Conv2d(10, 10, kernel_size=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            # 将四维的输出转成二维的输出，其形状为(批量大小, 10)
            nn.Flatten()


        )

    def forward(self, x):
        x = self.model(x)

        return x


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)



device = torch.device('cuda:0')
net=MLP().to(device)
net.apply(init_weights)
optimizer = optim.SGD(net.parameters(), lr=learning_rate)  # 梯度下降优化算法
criteon = nn.CrossEntropyLoss().to(device)

viz = Visdom()

viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
viz.line([[0.0, 0.0]], [0.], win='test', opts=dict(title='test loss&acc.',
                                                   legend=['loss', 'acc.']))
global_step = 0

for epoch in range(epochs):

    for batch_idx, (data, target) in enumerate(train_loader):  # 步指针，（张量数据，分类类别）来自于数据加载起（训练加载器）

        data, target = data.to(device), target.cuda()

        logits = net(data)
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()
        # print(w1.grad.norm(), w2.grad.norm())
        optimizer.step()

        global_step += 1
        viz.line([loss.item()], [global_step], win='train_loss', update='append')

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.cuda()
        logits = net(data)
        test_loss += criteon(logits, target).item()

        pred = logits.argmax(dim=1)
        correct += pred.eq(target).float().sum().item()

    viz.line([[test_loss, correct / len(test_loader.dataset)]],
             [global_step], win='test', update='append')
    viz.images(data.view(-1,1,224,224),win='x')
    viz.text(str(pred.detach().cpu().numpy()), win='pred',
             opts=dict(title='pred'))

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
