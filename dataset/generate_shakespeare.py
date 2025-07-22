import json
import os
import numpy as np
import random
import yaml

from dataset.utils.dataset_utils import check
from utils.language_utils import word_to_indices, letter_to_index

random.seed(1)
np.random.seed(1)
data_path_train = "../../shakespeare/data/train/all_data_niid_2_keep_0_train_9.json"
data_path_test = "../../shakespeare/data/test/all_data_niid_2_keep_0_test_9.json"


# https://github.com/TalwalkarLab/leaf/blob/master/models/shakespeare/stacked_lstm.py#L40
def process_x(raw_x_batch):
    x_batch = [word_to_indices(word) for word in raw_x_batch]
    x_batch = np.array(x_batch)
    return x_batch


def process_y(raw_y_batch):
    y_batch = [letter_to_index(c) for c in raw_y_batch]
    y_batch = np.array(y_batch)
    return y_batch

def generate_dataset(cfg):
    dir_path = cfg['dir_path']
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if check(cfg): return

    train_path = dir_path + "train/"
    test_path = dir_path + "test/"
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)

    with open(data_path_train) as f:
        raw_train = json.load(f)['user_data']
    with open(data_path_test) as f:
        raw_test  = json.load(f)['user_data']

    train_ = [{'x': process_x(v['x']), 'y': process_y(v['y'])} for v in raw_train.values()]
    test_ = [{'x': process_x(v['x']), 'y': process_y(v['y'])} for v in raw_test.values()]

    idx = sorted(range(len(train_)), key=lambda i: len(train_[i]['x']))
    train, test = [train_[i] for i in idx], [test_[i] for i in idx]

    for idx, data in enumerate(train):
        with open(f"{train_path}{idx}.npz", 'wb') as f:
            np.savez_compressed(f, data=data)

    for idx, data in enumerate(test):
        with open(f"{test_path}{idx}.npz", 'wb') as f:
            np.savez_compressed(f, data=data)


if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)
    generate_dataset(config)