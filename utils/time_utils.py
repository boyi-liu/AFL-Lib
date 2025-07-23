import time
import random

from functools import wraps
from utils.sys_utils import system_config


def time_record(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        self.training_time = execution_time * self.delay

        # downlink and uplink
        comm_time = self.comm_bytes() * 8 / (1024 * 1024) / self.bandwidth
        self.training_time += comm_time * 2

        dropout = system_config()['dropout']
        if random.random() < dropout['drop_prob']:
            self.training_time += (random.random() * dropout['drop_latency'])
        return result
    return wrapper