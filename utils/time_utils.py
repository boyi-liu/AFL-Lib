import time
import random

from functools import wraps

def time_record(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        self.training_time = execution_time * self.delay

        dropout = self.sysconfig['dropout']
        if dropout['dropout'] and random.random() < dropout['drop_prob']:
            self.training_time += (random.random() * dropout['drop_latency'])
        return result
    return wrapper