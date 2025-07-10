import numpy as np
import os
import random
import torch
import torchvision
import torchvision.transforms as transforms
import yaml

from utils.dataset_utils import check, separate_data, split_data, save_file

random.seed(1)
np.random.seed(1)

def generate_dataset(cfg):
    dir_path = cfg['dir_path']
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if check(cfg): return

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    trainset = torchvision.datasets.SVHN(root=os.path.join(dir_path, "rawdata"),
                                          split='train', download=True, transform=transform)
    testset = torchvision.datasets.SVHN(root=os.path.join(dir_path, "rawdata"),
                                         split='test', download=True, transform=transform)
    trainset.data, trainset.targets = next(
        iter(torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)))
    testset.data, testset.targets = next(
        iter(torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)))
    X = np.concatenate([trainset.data.numpy(), testset.data.numpy()])
    y = np.concatenate([trainset.labels.numpy(), testset.labels.numpy()])

    cfg['class_num'] = len(set(y))
    X, y, statistic = separate_data((X, y), cfg)
    train_data, test_data = split_data(X, y, cfg)
    save_file(train_data, test_data, cfg)


if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)
    generate_dataset(config)