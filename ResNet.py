import  torch
import  torch.nn as nn
import  torch.nn.functional as F
import  torch.optim as optim
from    torchvision import datasets, transforms  #torch数据预处理
from torchsummary import summary

from visdom import Visdom

batch_size=100
learning_rate=0.01
epochs=10

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([   #compose指将多个操作整合
                       transforms.ToTensor(),     #将其转化为数据范围在0到1的张量
                       # transforms.Normalize((0.1307,), (0.3081,))
                   ])),  #前面一个指令意为指定训练集
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True)  #batch_size表示批处理数据量，shuffle表示每次操作前打乱数据


datasets.ImageFolder()
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
                   )
summary(net, input_size=[(1,224,224)],batch_size=4,device="cpu")

def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
net.apply(init_weights)
optimizer = optim.SGD(net.parameters(), lr=learning_rate)#梯度下降优化算法

criteon = nn.CrossEntropyLoss().to(device)

viz = Visdom()

viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
viz.line([[0.0, 0.0]], [0.], win='test', opts=dict(title='test loss&acc.',
                                                   legend=['loss', 'acc.']))
global_step = 0

for epoch in range(epochs):

    for batch_idx, (data, target) in enumerate(train_loader):    #步指针，（张量数据，分类类别）来自于数据加载起（训练加载器）

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
    viz.images(data.view(-1, 1, 28, 28), win='x')
    viz.text(str(pred.detach().cpu().numpy()), win='pred',
             opts=dict(title='pred'))

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
torch.save(net.state_dict(),'ResNet.pkl')
