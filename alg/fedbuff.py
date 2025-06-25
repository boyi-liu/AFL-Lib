import torch

from alg.asyncbase import AsyncBaseClient, AsyncBaseServer, Status
from utils.time_utils import time_record


def add_args(parser):
    parser.add_argument('--etag', type=float, default=5)
    parser.add_argument('--k', type=int, default=10)
    return parser.parse_args()


class Client(AsyncBaseClient):
    @time_record
    def run(self):
        w_last = self.model2tensor()
        self.train()
        self.dW = self.model2tensor() -  w_last


class Server(AsyncBaseServer):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)
        self.buffer = []

    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()
        self.update_status()

    def aggregate(self):
        self.buffer.append(self.cur_client.dW)

        if len(self.buffer) == self.args.k:
            t_g = self.model2tensor()
            t_g_new = t_g + self.args.etag * torch.mean(torch.stack(self.buffer), dim=0) / self.args.k
            self.tensor2model(t_g_new)

            self.buffer = []