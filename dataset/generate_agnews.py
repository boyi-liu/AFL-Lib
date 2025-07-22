import numpy as np
import os
import random
import pandas as pd
import yaml

from utils.dataset_utils import check, separate_data, split_data, save_file
from utils.language_utils import tokenizer

random.seed(1)
np.random.seed(1)
max_len = 200
max_tokens = 32000

def generate_dataset(cfg):
    dir_path = cfg['dir_path']
    os.makedirs(dir_path, exist_ok=True)

    if check(cfg): return

    train_pat = dir_path + 'rawdata/train.csv'
    test_pat = dir_path + 'rawdata/test.csv'

    train_data = pd.read_csv(train_pat)
    test_data = pd.read_csv(test_pat)

    train_text = train_data['Title'].to_list()
    train_label = train_data['Class Index'].to_list()
    test_text = test_data['Title'].to_list()
    test_label = test_data['Class Index'].to_list()

    dataset_text = train_text + test_text
    dataset_label = train_label + test_label

    vocab, text_list = tokenizer(dataset_text, max_len, max_tokens)
    label_list = [int(l)-1 for l in dataset_label]

    text_lens = [len(text) for text in text_list]
    text_list = [(text, l) for text, l in zip(text_list, text_lens)]

    X = np.array(text_list, dtype=object)
    y = np.array(label_list)

    cfg['class_num'] = len(set(y))

    X, y, statistic = separate_data((X, y), cfg)
    train_data, test_data = split_data(X, y, cfg)
    save_file(train_data, test_data, cfg)


if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)
    generate_dataset(config)