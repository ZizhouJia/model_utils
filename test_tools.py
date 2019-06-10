import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


class cifar10_dataset(torch.utils.data.Dataset):
    def __init__(self,transform,train=True):
        self.dataset = torchvision.datasets.CIFAR10(root='./data', train=train,
                                           download=True, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
        data=list(self.dataset[index])
        data.insert(0,index)
        return tuple(data)



#in order for frame work test
def generate_cifar10_dataset(data_size=224,batch_size=32):
    transform = transforms.Compose(
    [transforms.Scale(data_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset =cifar10_dataset(transform=transform,train=True)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

    testset = cifar10_dataset(transform=transform,train=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
    return trainloader,testloader,testloader


def generate_optimizers(models,lr=0.1):
    return [torch.optim.SGD(models[0].parameters(),lr=lr,momentum=0.9)]


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        layer1 = nn.Sequential()
        layer1.add_module('conv1',nn.Conv2d(3,6,5))
        layer1.add_module('pool1',nn.MaxPool2d(2,2))
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv2',nn.Conv2d(6,16,5))
        layer2.add_module('pool2',nn.MaxPool2d(2,2))
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('fc1',nn.Linear(16*5*5,120))
        layer3.add_module('fc2',nn.Linear(120,84))
        layer3.add_module('fc3',nn.Linear(84,10))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0),-1)
        x = self.layer3(x)
        return x
