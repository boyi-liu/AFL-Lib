import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, class_num):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)

        self.fc = nn.Linear(256, class_num)

    def features(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc(x))
        return x

    def forward(self, x, return_feat=False):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        feat = x
        x = F.relu(self.fc(x))
        if return_feat:
            return x, feat
        return x

class MLP_HAR(nn.Module):
    def __init__(self, class_num, dim_in=900):
        super(MLP_HAR, self).__init__()
        self.fc1 = nn.Linear(dim_in, 300)
        self.fc2 = nn.Linear(300, class_num)

    def forward(self, x, return_feat=False):
        x = F.relu(self.fc1(x))
        feat = x
        x = self.fc2(x)
        if return_feat:
            return x, feat
        return x


def mlp_harbox(args):
    return MLP_HAR(class_num=args.class_num)

def mlp_mnist(args):
    return MLP(class_num=args.class_num)