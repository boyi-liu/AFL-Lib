import yaml
import numpy as np
import random

from dataset.utils.dataset_utils import split_data, save_file

random.seed(0)

cluster_set = ['hsh_', 'mmw_']
class_set = ['_walk', '_up', '_down']
label = [0, 1, 2]

NUM_OF_CLASS = 3
DIMENSION_OF_FEATURE = 900


def load_data(user_id):
    # dataset append and split

    if user_id < 4:
        cluster_id = 0
        intra_user_id = user_id + 1  # 0,1,2,3 to 1,2,3,4
    else:
        cluster_id = 1
        intra_user_id = user_id - 3  # 4,5,6 to 1,2,3

    # x append
    cluster_des = str(cluster_set[cluster_id])

    coll_class = []
    coll_label = []

    for class_id in range(NUM_OF_CLASS):
        read_path = './' + \
                    cluster_des + str(intra_user_id) + str(class_set[class_id]) + '_nor' + '.txt'

        temp_original_data = np.loadtxt(read_path, delimiter=',')
        temp_coll = temp_original_data.reshape(-1, DIMENSION_OF_FEATURE)
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
    for idx in range(7):
        dataset_train, dataset_test = load_data(idx)
        X.append(dataset_train)
        y.append(dataset_test)
    train_data, test_data = split_data(X, y, cfg)
    save_file(train_data, test_data, cfg)

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)
    generate_dataset(config)




