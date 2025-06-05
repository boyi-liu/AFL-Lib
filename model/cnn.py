import torch.nn as nn
import torch.nn.functional as F

class CNN_CIFAR(nn.Module):
    def __init__(self, class_num):
        super(CNN_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 64, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc = nn.Linear(64, class_num)

    def forward(self, x, return_feat=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        feat = x
        x = self.fc(x)
        if return_feat:
            return x, feat
        return x

def cnn_cifar10(args):
    return CNN_CIFAR(class_num=args.class_num)

def cnn_cifar100(args):
    return CNN_CIFAR(class_num=args.class_num)