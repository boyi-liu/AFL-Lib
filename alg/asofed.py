from alg.asyncbase import AsyncBaseClient, AsyncBaseServer
from utils.time_utils import time_record
import torch


def add_args(parser):
    parser.add_argument('--lam', type=float, default=0.01)
    parser.add_argument('--beta', type=float, default=0.5)
    return parser.parse_args()


class Client(AsyncBaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.prev_s_grad = None  
        self.h = None
        self.v = None
        self.lam = args.lam
        self.beta = args.beta


    @time_record
    def run(self):
        self.train()

    def train(self):
        w_global = self.model2tensor()
        
        total_loss = 0.0
        for epoch in range(self.epoch):
            for idx, data in enumerate(self.loader_train):
                X, y = self.preprocess(data)
                preds = self.model(X)
                loss = self.loss_func(preds, y)
               
                w_local = self.model2tensor()
                loss += (self.lam / 2) * torch.norm(w_local - w_global, p=2)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                total_loss += loss.item()
        
        self.metric['loss'] = total_loss / len(self.loader_train)

        # Line 11 - Compute s_grad
        s_grad = w_global - self.model2tensor()

        # Line 12 - set h_prev
        if self.h is None: self.h = torch.zeros_like(s_grad)
        prev_h = self.h

        # Line 13 - solve zeta_grad
        if self.prev_s_grad is None: self.prev_s_grad = torch.zeros_like(s_grad)
        self.zeta_grad = s_grad - self.prev_s_grad + prev_h

        # None need for Line 14, because we directly upload zeta_grad

        # Line 15 - update h
        self.h = self.beta * self.h + (1-self.beta) * self.prev_s_grad

        # Line 16 - update v
        self.prev_s_grad = s_grad.clone().detach()


class Server(AsyncBaseServer):
    def aggregate(self):
        # update w
        zeta = self.cur_client.zeta_grad
        model_g = self.model2tensor()
        model_g -= (zeta * len(self.cur_client.dataset_train)) / sum(len(c.dataset_train) for c in self.clients)
        self.tensor2model(model_g)

        # update w with feature learning
        # def feature_update(W):
        #     alpha = torch.softmax(torch.abs(W), dim=0)
        #     return alpha * W
        # with torch.no_grad():
        #     W = next(self.model.parameters())
        #     W.copy_(feature_update(W))


    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()
        self.update_status()