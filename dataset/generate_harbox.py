import numpy as np
import yaml
import os
import random

from dataset.utils.dataset_utils import save_file, split_data

random.seed(0)

class_set = ['Call', 'Hop', 'typing', 'Walk', 'Wave']
label = [0, 1, 2, 3, 4]

NUM_OF_CLASS = 5
DIMENSION_OF_FEATURE = 900

SOURCE_PATH = '../../large_scale_HARBox'

def load_data(client_id):
    coll_class = []
    coll_label = []

    for class_id in range(NUM_OF_CLASS):
        # NOTE: the client-0 is empty!
        read_path = SOURCE_PATH + '/' + str(client_id+1) + '/' + str(class_set[class_id]) + '_train' + '.txt'
        print(read_path)

        if os.path.exists(read_path):
            temp_original_data = np.loadtxt(read_path)
            temp_reshape = temp_original_data.reshape(-1, 100, 10)
            temp_coll = temp_reshape[:, :, 1:10].reshape(-1, DIMENSION_OF_FEATURE)
            count_img = temp_coll.shape[0]
            temp_label = class_id * np.ones(count_img)
            coll_class.extend(temp_coll)
            coll_label.extend(temp_label)

    coll_class = np.array(coll_class)
    coll_label = np.array(coll_label)

    return coll_class, coll_label

def generate_dataset(cfg):
    X = []
    y = []
    for idx in range(1):
        dataset_train, dataset_test = load_data(idx)
        X.append(dataset_train)
        y.append(dataset_test)
    train_data, test_data = split_data(X, y, cfg)
    save_file(train_data, test_data, cfg)

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)
    generate_dataset(config)