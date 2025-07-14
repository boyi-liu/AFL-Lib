from alg.asyncbase import AsyncBaseClient, AsyncBaseServer, Status
from utils.time_utils import time_record

import torch


def add_args(parser):
    parser.add_argument('--M', type=int, default=10, help='buffer size M for CA2FL')
    parser.add_argument('--eta', type=float, default=0.01, help='gloal lr')
    return parser.parse_args()


class Client(AsyncBaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        
    @time_record
    def run(self):
        w_last = self.model2tensor()
        self.train()
        self.dW = self.model2tensor() - w_last


class Server(AsyncBaseServer):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)
        self.buffer = []
        self.buffer_clients = []
        self.h_cache = [torch.zeros_like(self.model2tensor()) for _ in self.clients]
        self.momentum_cache = torch.zeros_like(self.model2tensor())
        self.M = args.M
        self.eta = args.eta

    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()
        self.update_status()

    def aggregate(self):
        c_id = self.cur_client.id
        self.buffer.append(self.cur_client.dW - self.h_cache[c_id])
        self.h_cache[c_id] = self.cur_client.dW

        if len(self.buffer) == self.M:
            h_t = torch.mean(torch.stack(self.h_cache), dim=0)
            calibration = sum(delta - self.h_cache[cid] for delta, cid in zip(self.buffer, self.buffer_clients)) / len(
                self.buffer)
            v_t = h_t + calibration

            t_g = self.model2tensor()
            t_g_new = t_g + self.eta * v_t
            self.tensor2model(t_g_new)

    def update_status(self):
        # set the current client to idle
        self.cur_client.status = Status.IDLE

        # update the staleness
        if len(self.buffer) >= self.M:
            for c in filter(lambda x: x.status == Status.ACTIVE, self.clients):
                self.staleness[c.id] += 1
            self.buffer.clear()
            self.buffer_clients.clear()