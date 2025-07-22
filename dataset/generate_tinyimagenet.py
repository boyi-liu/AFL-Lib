import numpy as np
import os
import random
import torch
import torchvision.transforms as transforms
import yaml

from utils.dataset_utils import check, separate_data, split_data, save_file
from torchvision.datasets import ImageFolder, DatasetFolder

random.seed(1)
np.random.seed(1)

# https://github.com/QinbinLi/MOON/blob/6c7a4ed1b1a8c0724fa2976292a667a828e3ff5d/datasets.py#L148
class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)

def generate_dataset(cfg):
    dir_path = cfg['dir_path']
    os.makedirs(dir_path, exist_ok=True)

    if check(cfg): return

    if not os.path.exists(f'{dir_path}/rawdata/'):
        os.system(f'wget --directory-prefix {dir_path}/rawdata/ http://cs231n.stanford.edu/tiny-imagenet-200.zip')
        os.system(f'unzip {dir_path}/rawdata/tiny-imagenet-200.zip -d {dir_path}/rawdata/')
    else:
        print('rawdata already exists.\n')

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    trainset = ImageFolder_custom(root=dir_path + 'rawdata/tiny-imagenet-200/train/', transform=transform)
    testset = ImageFolder_custom(root=dir_path + 'rawdata/tiny-imagenet-200/val/', transform=transform)

    trainset.data, trainset.targets = next(
        iter(torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)))
    testset.data, testset.targets = next(
        iter(torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)))

    X = np.concatenate([trainset.data.numpy(), testset.data.numpy()])
    y = np.concatenate([trainset.targets.numpy(), testset.targets.numpy()])

    cfg['class_num'] = len(set(y))
    X, y, statistic = separate_data((X, y), cfg)
    train_data, test_data = split_data(X, y, cfg)
    save_file(train_data, test_data, cfg)


if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)
    generate_dataset(config)