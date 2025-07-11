import torch

from alg.asyncbase import AsyncBaseClient, AsyncBaseServer
from utils.time_utils import time_record


def add_args(parser):
    parser.add_argument('--drop_rate', type=float, default=0.5, help='dropout rate')
    return parser.parse_args()


class Client(AsyncBaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.mask = {name: torch.ones_like(param, dtype=torch.int)
                     for name, param in self.model.named_parameters()}
        self.drop_rate = args.drop_rate

    @time_record
    def run(self):
        self.train()

    def train(self):
        # === train ===
        total_loss = 0.0
        self.apply_mask()
        for epoch in range(self.epoch):
            for data in self.loader_train:
                X, y = self.preprocess(data)
                preds = self.model(X)
                loss = self.loss_func(preds, y)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                self.apply_mask()
                total_loss += loss.item()

        # === record loss ===
        self.metric['loss'] = total_loss / len(self.loader_train)

    def apply_mask(self):
        for name, param in self.model.named_parameters():
            param.data *= self.mask[name]

    def generate_mask(self):
        for name, param in self.model.named_parameters():
            num_elements = param.numel()
            num_zeros = int(num_elements * self.drop_rate)
            num_ones = num_elements - num_zeros

            flat_mask = torch.cat([
                torch.zeros(num_zeros),
                torch.ones(num_ones)
            ])

            perm = torch.randperm(num_elements)
            flat_mask = flat_mask[perm]

            self.mask[name] = flat_mask.view_as(param)

class Server(AsyncBaseServer):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)
        self.decay = args.decay

    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()
        self.update_status()

    def aggregate(self):
        decay = self.decay

        for mask, param, c_param in zip(self.cur_client.mask.values(), self.model.parameters(), self.cur_client.model.parameters()):
            mask_c = torch.ones_like(mask, dtype=torch.int) - mask
            param.data = param.data * mask_c + ((1-decay) * param.data + decay * c_param.data) * mask