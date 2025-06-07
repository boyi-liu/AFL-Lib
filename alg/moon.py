import torch
import torch.nn.functional as F

from alg.base import BaseClient, BaseServer
from utils.time_utils import time_record

def add_args(parser):
    parser.add_argument('--tau', type=float, default=0.5, help="Temperature")
    parser.add_argument('--mu', type=float, default=1, help="Mu")
    return parser.parse_args()


class Client(BaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.tau = args.tau
        self.mu = args.mu
        self.prev_m = None

    @time_record
    def run(self):
        self.train()

    def train(self):
        gm = self.model2tensor() # this is only param.data, without grad
        total_loss = 0.0
        batch_loss = []
        for epoch in range(self.epoch):
            for idx, data in enumerate(self.loader_train):
                X, y = self.preprocess(data)

                pm = torch.cat([param.view(-1) for param in self.model.parameters()], dim=0)

                if self.prev_m is not None:
                    # output for global model
                    self.tensor2model(gm)
                    with torch.no_grad():
                        _, rep_g = self.model(X, return_feat=True)
                        rep_g = rep_g.detach()

                    # output for prev model
                    self.tensor2model(self.prev_m)
                    with torch.no_grad():
                        _, rep_prev = self.model(X, return_feat=True)
                        rep_prev = rep_prev.detach()

                # output for current model
                self.tensor2model(pm)
                preds, rep = self.model(X, return_feat=True)

                loss = self.loss_func(preds, y)
                if self.prev_m is not None:
                    dis_global = F.cosine_similarity(rep, rep_g) / self.tau
                    dis_prev = F.cosine_similarity(rep, rep_prev) / self.tau
                    loss_con = - torch.log(torch.exp(dis_global) / (torch.exp(dis_global) + torch.exp(dis_prev)))
                    loss += self.mu * torch.mean(loss_con)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                total_loss += loss.item()
        self.metric['loss'] = total_loss / len(self.loader_train)
        self.prev_m = self.model2tensor()
        del gm

class Server(BaseServer):
    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()