import os
import random
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

def check(cfg):
    config_path = cfg['dir_path'] + "/config.yaml"
    train_path = cfg['dir_path'] + "/train/"
    test_path = cfg['dir_path'] + "/test/"

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.load(f.read(), Loader=yaml.Loader)
        if config == cfg:
            print("\nDataset already generated.\n")
            return True

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return False


def separate_data(data, cfg):
    num_clients = cfg['client_num']
    partition = cfg['partition']
    class_num = cfg['class_num']
    batch_size = cfg['batch_size']
    train_ratio = cfg['train_ratio']
    class_per_client = cfg['class_per_client']
    iid_proportion = cfg['iid_proportion']
    group_num = cfg['group_num']

    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data
    least_samples = int(min(batch_size / (1 - train_ratio), len(dataset_label) / num_clients / 2))

    dataidx_map = {}

    if partition == 'iid':
        partition = 'pat'
        class_per_client = class_num

    if partition == 'pat':
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = [idxs[dataset_label == i] for i in range(class_num)]
        class_num_per_client = [class_per_client for _ in range(num_clients)]

        for i, class_idxs in enumerate(idx_for_each_class):
            selected_clients = [c for c, n in enumerate(class_num_per_client) if n > 0]
            selected_clients = selected_clients[:int(np.ceil(num_clients / class_num * class_per_client))]

            num_total = len(class_idxs)
            num_clients_sel = len(selected_clients)
            sizes = [num_total // num_clients_sel] * (num_clients_sel - 1)
            sizes.append(num_total - sum(sizes))

            idx = 0
            for client, size in zip(selected_clients, sizes):
                dataidx_map.setdefault(client, np.array([], dtype=int))
                dataidx_map[client] = np.concatenate([dataidx_map[client], class_idxs[idx:idx+size]])
                class_num_per_client[client] -= 1
                idx += size

    elif partition == "dir":
        min_size = 0
        N = len(dataset_label)
        try_cnt = 1

        while min_size < least_samples:
            if try_cnt > 1:
                print(f"Try {try_cnt}: Some client has fewer than {least_samples} samples. Retrying...")

            idx_batch = [[] for _ in range(num_clients)]
            for k in range(class_num):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)

                proportions = np.random.dirichlet([cfg['alpha']] * num_clients)
                mask = np.array([len(b) < N / num_clients for b in idx_batch])
                proportions = proportions * mask
                proportions = proportions / proportions.sum()

                split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                split_idxs = np.split(idx_k, split_points)
                idx_batch = [b + s.tolist() for b, s in zip(idx_batch, split_idxs)]
            min_size = min(len(b) for b in idx_batch)
            try_cnt += 1

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]

    elif partition == 'group':
        idxs = np.arange(len(dataset_label))
        idx_for_each_class = [idxs[dataset_label == i] for i in range(class_num)]

        # ========== Step 1: s% uniform allocation ==========
        for i, idx_class in enumerate(idx_for_each_class):
            np.random.shuffle(idx_class)
            selected_clients = list(range(num_clients))
            num_all = int(len(idx_class) * iid_proportion)
            num_clients_sel = len(selected_clients)

            base = num_all // num_clients_sel
            num_samples = [base] * num_clients_sel
            num_samples[random.randint(0, num_clients_sel - 1)] += num_all - sum(num_samples)

            idx = 0
            for client, n in zip(selected_clients, num_samples):
                dataidx_map.setdefault(client, np.array([], dtype=int))
                dataidx_map[client] = np.concatenate([dataidx_map[client], idx_class[idx:idx + n]])
                idx += n

            idx_for_each_class[i] = idx_class[idx:]

            # ========== Step 2: (1 - s)% group allocation ==========
            def split_into_groups(total, num_groups):
                sizes = [total // num_groups] * (num_groups - 1)
                sizes.append(total - sum(sizes))
                indices = []
                start = 0
                for size in sizes:
                    end = start + size
                    indices.append(list(range(start, end)))
                    start = end
                return indices

            class_groups = split_into_groups(class_num, group_num)
            client_groups = split_into_groups(num_clients, group_num)

            for class_group, client_group in zip(class_groups, client_groups):
                for i in class_group:
                    idx_class = idx_for_each_class[i]
                    np.random.shuffle(idx_class)
                    num_all = len(idx_class)
                    num_clients_sel = len(client_group)

                    base = num_all // num_clients_sel
                    num_samples = [base] * num_clients_sel
                    num_samples[random.randint(0, num_clients_sel - 1)] += num_all - sum(num_samples)

                    idx = 0
                    for client, n in zip(client_group, num_samples):
                        dataidx_map.setdefault(client, np.array([], dtype=int))
                        dataidx_map[client] = np.concatenate([dataidx_map[client], idx_class[idx:idx + n]])
                        idx += n
    else:
        raise NotImplementedError

    # assign data
    for client in range(num_clients):
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]

        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client] == i))))

    del data

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    return X, y, statistic


def split_data(X, y, cfg):
    train_ratio = cfg['train_ratio']
    train_data, test_data = [], []
    train_counts, test_counts = [], []

    for x_i, y_i in zip(X, y):
        x_train, x_test, y_train, y_test = train_test_split(x_i, y_i, train_size=train_ratio, shuffle=True)
        train_data.append({'x': x_train, 'y': y_train})
        test_data.append({'x': x_test, 'y': y_test})
        train_counts.append(len(y_train))
        test_counts.append(len(y_test))

    print(f"Total number of samples: {sum(train_counts + test_counts)}")
    print(f"The number of train samples: {train_counts}")
    print(f"The number of test samples: {test_counts}\n")
    del X, y

    return train_data, test_data


def save_file(train_data, test_data, config):
    dir_path = config['dir_path']
    config_path = f"{dir_path}/config.yaml"
    train_path = f"{dir_path}/train/"
    test_path = f"{dir_path}/test/"

    print("Saving to disk.\n")

    for idx, train_dict in enumerate(train_data):
        with open(train_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(test_data):
        with open(test_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_dict)
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    print("Finish generating dataset.\n")