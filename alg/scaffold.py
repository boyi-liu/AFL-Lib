from alg.base import BaseClient, BaseServer
from utils.time_utils import time_record

import torch

def add_args(parser):
    parser.add_argument('--eta_g', type=float, default=0.001)
    return parser.parse_args()

class Client(BaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.C = torch.zeros_like(self.model2tensor())
        self.delta_C = torch.zeros_like(self.model2tensor())
        self.delta_W = torch.zeros_like(self.model2tensor())

    @time_record
    def run(self):
        prev_model = self.model2tensor()
        self.train()
        self.tensor2model(self.model2tensor() - self.lr * (self.server.C - self.C))
        C_plus = self.C - self.server.C + (prev_model - self.model2tensor()) / (self.epoch * self.lr)
        self.delta_C = C_plus - self.C
        self.C = C_plus
        self.delta_W = self.model2tensor() - prev_model


class Server(BaseServer):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)
        self.C = torch.zeros_like(self.model2tensor())
        self.eta_g = args.eta_g

    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()

    def uplink(self):
        assert (len(self.sampled_clients) > 0)
        def nan_to_zero(tensor):
            return torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
        self.received_params = [nan_to_zero(client.delta_W) for client in self.sampled_clients]
        self.received_C = [nan_to_zero(client.delta_C) for client in self.sampled_clients]

    def aggregate(self):
        assert (len(self.sampled_clients) > 0)
        delta_C_avg = sum(self.received_C) / len(self.received_C)
        delta_W_avg = sum(self.received_params) / len(self.received_params)

        self.tensor2model(self.model2tensor() + self.eta_g * delta_W_avg)
        self.C += delta_C_avg * len(self.sampled_clients) / self.client_num
