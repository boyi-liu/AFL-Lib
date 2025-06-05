import heapq
import random
import numpy as np

from alg.base import BaseClient, BaseServer
from enum import Enum

# Usage of Status
# During training, those training are set to Status.ACTIVE, the active clients will update staleness, and will not be sampled
# After aggregation, the aggregated is set to Status.IDLE

class Status(Enum):
    IDLE = 1
    ACTIVE = 2


class AsyncBaseClient(BaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.status = Status.IDLE

    def reset_optimizer(self, decay=True):
        if not decay:
            return
        for param_group in self.optim.param_groups:
            param_group['lr'] =  self.lr * (self.args.gamma ** (self.server.round / int(self.args.total_num * self.args.sr)))

    def run(self):
        raise NotImplementedError


class AsyncBaseServer(BaseServer):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)
        self.decay = args.decay

        self.MAX_CONCURRENCY = int(self.client_num * self.sample_rate)
        self.client_queue = []
        self.staleness = [0 for _ in self.clients]
        self.cur_client = None

    def run(self):
        raise NotImplementedError

    def sample(self):
        active_num = len([c for c in self.clients if c.status == Status.ACTIVE])
        if active_num >= self.MAX_CONCURRENCY:
            return

        idle_clients = [c for c in self.clients if c.status != Status.ACTIVE]
        self.sampled_clients = random.sample(idle_clients, self.MAX_CONCURRENCY - active_num)
        for c in self.sampled_clients: self.staleness[c.id] = 0

    def downlink(self):
        for c in filter(lambda x: x.status != Status.ACTIVE, self.sampled_clients):
            c.clone_model(self)

    def client_update(self):
        for c in filter(lambda x: x.status != Status.ACTIVE, self.sampled_clients):
            c.model.train()
            c.reset_optimizer(True)
            c.run()
            heapq.heappush(self.client_queue, (self.wall_clock_time + c.training_time, c))
            c.status = Status.ACTIVE

    def uplink(self):
        self.wall_clock_time, self.cur_client = heapq.heappop(self.client_queue)

    def aggregate(self):
        t_aggr = self.decay * self.cur_client.model2tensor() + (1 - self.decay) * self.model2tensor()
        self.tensor2model(t_aggr)

    def update_status(self):
        # set the current client to idle
        self.cur_client.status = Status.IDLE

        # update the staleness
        for c in filter(lambda x: x.status == Status.ACTIVE, self.clients):
            self.staleness[c.id] += 1

    def test_all(self):
        self.metric['acc'] = []
        for client in self.clients:
            # NOTE: have to store current local model
            context = client.model2tensor()
            client.clone_model(self)
            client.local_test()
            client.tensor2model(context)
            self.metric['acc'].append(client.metric['acc'])

        return {
            'acc': np.mean(self.metric['acc']),
            'acc_std': np.std(self.metric['acc']),
        }