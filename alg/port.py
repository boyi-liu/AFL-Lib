import heapq
import torch

from alg.asyncbase import AsyncBaseClient, AsyncBaseServer, Status
from enum import Enum
from utils.time_utils import time_record

def add_args(parser):
    parser.add_argument('--alpha', type=float, default=3.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--omega', type=int, default=10)
    parser.add_argument('--min_clients', type=int, default=10)
    return parser.parse_args()

class EventStatus(Enum):
    READY = 0
    TRAIN = 1
    AGGREGATE = 2

class Event:
    def __init__(self, status, client):
        self.status = status
        self.client = client

    def __lt__(self, other):
        return self.client.id < other.client.id


class Client(AsyncBaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.cur_epoch = 0
        self.urgent = False

    @time_record
    def run(self):
        self.train_1_epoch()

    def train_1_epoch(self):
        for data in self.loader_train:
            X, y = self.preprocess(data)
            preds = self.model(X)
            loss = self.loss_func(preds, y)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()


class Server(AsyncBaseServer):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)
        self.alpha = args.alpha
        self.beta = args.beta
        self.omega = args.omega
        self.min_clients = args.min_clients
        self.prev_model = None
        self.buffer = []

    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.aggregate()
        self.notify()
        self.update_status()

    def client_update(self):
        for c in filter(lambda x: x.status != Status.ACTIVE, self.sampled_clients):
            heapq.heappush(self.client_queue, (self.wall_clock_time, Event(EventStatus.READY, c)))

        while True:
            self.wall_clock_time, event = heapq.heappop(self.client_queue)
            if event.status == EventStatus.AGGREGATE:
                self.cur_client = event.client
                break
            else:
                c = event.client
                c.cur_epoch = 0 if event.status == EventStatus.READY else c.cur_epoch + 1

                if c.cur_epoch == self.epoch or c.urgent:
                    heapq.heappush(self.client_queue, (self.wall_clock_time, Event(EventStatus.AGGREGATE, c)))
                else:
                    c.model.train()
                    c.reset_optimizer(False)
                    c.run()
                    heapq.heappush(self.client_queue,
                                   (self.wall_clock_time + c.training_time, Event(EventStatus.TRAIN, c)))
                c.status = Status.ACTIVE

    def update_status(self):
        # set the current client to idle
        self.cur_client.status = Status.IDLE

        # update the staleness
        if len(self.buffer) >= self.min_clients:
            for c in filter(lambda x: x.status == Status.ACTIVE, self.clients):
                self.staleness[c.id] += 1
            self.buffer = []

    def compute_staleness_discount(self, c_id):
        staleness = self.staleness[c_id]
        return self.alpha * self.omega / (staleness + self.omega)

    def compute_interference_discount(self, c_update):
        if self.prev_model is None:
            return self.beta

        delta_g = self.model2tensor() - self.prev_model
        cosine_sim = torch.cosine_similarity(c_update.view(-1), delta_g.view(-1), dim=0)
        return self.beta * (cosine_sim + 1) / 2

    def aggregate(self):
        self.buffer.append({
            'id': self.cur_client.id,
            'update': self.cur_client.model2tensor(),
            'size': len(self.cur_client.dataset_train),
        })

        if len(self.buffer) >= self.min_clients:
            weights = []
            for b in self.buffer:
                s_discount = self.compute_staleness_discount(b['id'])
                i_discount = self.compute_interference_discount(b['update'])
                weights.append(b['size'] * (s_discount + i_discount))
            weights = torch.tensor(weights) / torch.sum(torch.tensor(weights))
            aggr_tensor = sum(b['update'] * w for b, w in zip(self.buffer, weights))
            self.tensor2model(aggr_tensor)
            self.prev_model = aggr_tensor

    def notify(self):
        for client in filter(lambda x: x.status == Status.ACTIVE, self.clients):
            if self.staleness[client.id] > self.omega: client.urgent = True