from alg.asyncbase import AsyncBaseClient, AsyncBaseServer, Status
from utils.time_utils import time_record
import torch


def add_args(parser):
    return parser.parse_args()


class Client(AsyncBaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.p = len(self.dataset_train)
        
    @time_record
    def run(self):
        self.train()


class Server(AsyncBaseServer):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)
        self.v = 0
        self.alpha_sum = [0 for _ in self.clients]
        self.data_ratios = [c.data_volume / sum(c.data_volume for c in clients) for c in clients]

    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()
        self.update_status()

    def aggregate(self):
        client_id = self.cur_client.id
        d_i = self.data_ratios[client_id]
        alpha_sum_i = self.alpha_sum[client_id]
        alpha_i = min(1, d_i / len(self.clients) * (self.v + 1) - alpha_sum_i)

        t_aggr = alpha_i * self.cur_client.model2tensor() + (1 - alpha_i) * self.model2tensor()
        self.tensor2model(t_aggr)

        self.v += 1
        self.alpha_sum[client_id] += alpha_i