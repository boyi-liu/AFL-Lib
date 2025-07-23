import random

import numpy as np
import yaml

SCALE_FACTOR = 50

# Training time of Jetson TX2, Jetson Nano, Raspberry Pi
# `Benchmark Analysis of Jetson TX2, Jetson Nano and Raspberry PI using Deep-CNN`
device_reference = [1, 1.8125, 11.625]

# WiFi, 150-600 Mbps
# 4G, 20-100 Mbps
# 5G, 50-1000 Mbps
bandwidths = [(150, 600), (20, 100), (50, 1000)]


def system_config():
    with open('utils/sys.yaml', 'r') as f:
        sys_config = yaml.load(f.read(), Loader=yaml.Loader)
    return sys_config

def probs_to_counts(probs, total_count):
    raw_counts = np.array(probs) * total_count
    floored = np.floor(raw_counts).astype(int)
    remainder = total_count - floored.sum()

    fractional_parts = raw_counts - floored
    indices = np.argsort(-fractional_parts)

    for i in range(remainder):
        floored[indices[i]] += 1

    return floored.tolist()

def device_config(client_num):
    sys_config = system_config()
    prop = sys_config['dev']['dev_prop']
    prop = list(map(float, prop.split(' ')))
    prop = [p / sum(prop) for p in prop]

    counts = probs_to_counts(prop, client_num)
    result = [val for val, count in zip(device_reference, counts) for _ in range(count)]
    random.shuffle(result)
    return [r * SCALE_FACTOR for r in result]

    group_sizes = np.round(np.array(prop) * client_num).astype(int)
    group_sizes[-1] += client_num - group_sizes.sum()

    device_time = np.repeat([d * SCALE_FACTOR for d in device_reference], group_sizes)

    return device_time[id]

def comm_config(client_num):
    sys_config = system_config()
    comm = sys_config['comm']['comm']
    if not comm: return [0 for _ in range(client_num)]

    prop = sys_config['comm']['comm_prop']
    prop = list(map(float, prop.split(' ')))
    prop = [p / sum(prop) for p in prop]

    counts = probs_to_counts(prop, client_num)
    result = [val for val, count in zip(bandwidths, counts) for _ in range(count)]
    random.shuffle(result)
    return [random.uniform(min_bandwidth, max_bandwidth) for min_bandwidth, max_bandwidth in result]

    min_bandwidth, max_bandwidth = random.choices(bandwidths, weights=prop, k=1)[0]
    bandwidth = random.uniform(min_bandwidth, max_bandwidth)

    return calculate_model_size(model) * 8 / bandwidth

def calculate_model_size(model):
    total_size = 0
    for name, param in model.named_parameters():
        total_size += param.numel() * param.element_size()
    return total_size / (1024 * 1024)