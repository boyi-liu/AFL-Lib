import random

import numpy as np
import yaml

SCALE_FACTOR = 50

# Training time of TX2, Nano, Pi
# `Benchmark Analysis of Jetson TX2, Jetson Nano and Raspberry PI using Deep-CNN`
device_reference = [1, 1.8125, 11.625]

# 4G, 20Mbps
# WiFi, 100Mbps
# 5G, 500Mbps
bandwidths = [20, 100, 500]


def system_config():
    with open('utils/sys.yaml', 'r') as f:
        sys_config = yaml.load(f.read(), Loader=yaml.Loader)
    return sys_config

def device_config(id, client_num):
    sys_config = system_config()
    prop = sys_config['dev']['dev_prop']
    prop = list(map(float, prop.split(' ')))

    # normalize
    prop = [p / sum(prop) for p in prop]

    group_sizes = np.round(np.array(prop) * client_num).astype(int)
    group_sizes[-1] += client_num - group_sizes.sum()

    device_time = np.repeat([d * SCALE_FACTOR for d in device_reference], group_sizes)

    return device_time[id]

def comm_config(model):
    sys_config = system_config()

    comm = sys_config['comm']['comm']
    if not comm: return 0

    prop = sys_config['comm']['comm_prop']
    prop = list(map(float, prop.split(' ')))

    # normalize
    prop = [p / sum(prop) for p in prop]

    bandwidth = random.choices(bandwidths, weights=prop, k=1)[0]

    return calculate_model_size(model) * 8 / bandwidth

def calculate_model_size(model):
    total_size = 0
    for name, param in model.named_parameters():
        total_size += param.numel() * param.element_size()
    return total_size / (1024 * 1024)