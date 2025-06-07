import torch

from utils.time_utils import time_record
from alg.base import BaseClient, BaseServer

def add_args(parser):
    parser.add_argument('--mu', type=float, default=0.05, help="Mu")
    return parser.parse_args()

class Client(BaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.mu = args.mu

    @time_record
    def run(self):
        self.train()

    def train(self):
        gm = self.model2tensor() # this is only param.data, without grad
        total_loss = 0.0
        for epoch in range(self.epoch):
            for idx, data in enumerate(self.loader_train):
                X, y = self.preprocess(data)
                preds = self.model(X)

                pm = torch.cat([param.view(-1) for param in self.model.parameters()], dim=0)
                loss = self.loss_func(preds, y)
                loss += self.mu * torch.norm(gm - pm, p=2)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                total_loss += loss.item()
        self.metric['loss'] = total_loss / len(self.loader_train)


class Server(BaseServer):
    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()