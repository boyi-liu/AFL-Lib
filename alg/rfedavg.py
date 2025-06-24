import torch

from utils.time_utils import time_record
from trainer.base import BaseServer, BaseClient

def mmd(X, Y):
    X_avg = torch.sum(X, dim=0) / X.shape[0]
    Y_avg = torch.sum(Y, dim=0) / Y.shape[0]
    return torch.norm(X_avg - Y_avg) ** 2

def add_args(parser):
    parser.add_argument('--lam', type=float, default=1e-5, help="Lambda")
    return parser.parse_args()

class Client(BaseClient):
    def __init__(self, id, args, dataset):
        super().__init__(id, args, dataset)
        self.features = []
        self.lam = args.lam

    @time_record
    def run(self):
        self.train()

    def train(self):
        total_loss = 0.0
        for epoch in range(self.epoch):
            for idx, data in enumerate(self.loader_train):
                X, y = self.preprocess(data)
                preds, feat = self.model(X, return_feat=True)
                mmd_loss = sum([mmd(feat, f_t) for f_t in self.features]) / len(self.features)
                
                loss = self.loss_func(preds, y)
                loss += self.lam * mmd_loss

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                total_loss += loss.item()
        self.metric['loss'] = total_loss / len(self.loader_train)

    def get_features(self):
        inputs, _ = next(iter(self.loader_train))
        inputs = inputs.to(self.device)
        return self.model.features(inputs)


class Server(BaseServer):
    def __init__(self, id, args, dataset, clients):
        super().__init__(id, args, dataset, clients)
        self.features = []

    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()

    def cal_features(self):
        self.features = [client.get_features().detach() for client in self.sampled_clients]

    def downlink(self):
        assert (len(self.sampled_clients) > 0)
        for i, client in enumerate(self.sampled_clients):
            client.clone_model(self)
            client.features = [item for j, item in enumerate(self.features) if i != j]