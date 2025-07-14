import torch
from alg.asyncbase import AsyncBaseClient, AsyncBaseServer, Status
from utils.time_utils import time_record


def add_args(parser):
    parser.add_argument('--beta1', type=float, default=0.6)
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--buffer_size', type=int, default=20)
    parser.add_argument('--eta_g', type=float, default=1)
    return parser.parse_args()


class Client(AsyncBaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.C = torch.zeros_like(self.model2tensor())

    def train(self):
        # === train ===
        total_loss = 0.0

        for epoch in range(self.epoch):
            for data in self.loader_train:
                X, y = self.preprocess(data)
                preds = self.model(X)
                loss = self.loss_func(preds, y)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                cur_model = self.model2tensor() - self.lr * (self.server.C - self.C)
                self.tensor2model(cur_model)

                total_loss += loss.item()

        # === record loss ===
        self.metric['loss'] = total_loss / len(self.loader_train)

    @time_record
    def run(self):
        self.prev_model = self.model2tensor()

        self.train()

        cur_model = self.model2tensor()
        hat_C = ((self.prev_model - cur_model) - (self.server.C - self.C)) / (self.epoch * self.lr)
        self.dC = hat_C - self.C
        self.C = hat_C

        self.dW = cur_model - self.prev_model


class Server(AsyncBaseServer):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)
        self.C = torch.zeros_like(self.model2tensor())
        self.m = torch.zeros_like(self.model2tensor())
        self.v = torch.zeros_like(self.model2tensor())


        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.eta_g = args.eta_g

        self.buffer_size = args.buffer_size

        self.dW_buffer = []
        self.dC_buffer = []
        self.prev_model_buffer = []

        self.epsilon = 1e-8

    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()
        self.update_status()

    def aggregate(self):
        self.dW_buffer.append(self.cur_client.dW)
        self.dC_buffer.append(self.cur_client.dC)
        self.prev_model_buffer.append(self.cur_client.prev_model)

        if len(self.dW_buffer) >= self.buffer_size:
            # compute w_i
            r_list = []
            for dW, prev_model in zip(self.dW_buffer, self.prev_model_buffer):
                delta_global = (self.model2tensor() - prev_model) if self.round > self.buffer_size else self.model2tensor()
                r = torch.nn.functional.cosine_similarity(delta_global, dW, dim=0)
                r_list.append(max(r, 0))
            w_list = [r / (sum(r_list) + self.epsilon) for r in r_list]

            # update model
            dW = sum([w * dW for w, dW in zip(w_list, self.dW_buffer)])

            self.m = self.beta1 * self.m + (1 - self.beta1) * dW
            self.v = self.beta2 * self.v + (1 - self.beta2) * (dW ** 2)

            m_hat = self.beta1 * self.m + (1 - self.beta1) * dW
            model_g = self.model2tensor() + self.eta_g * m_hat / (torch.sqrt(self.v) + self.epsilon)

            self.tensor2model(model_g)

            # update c
            self.C += sum([w * dC for w, dC in zip(w_list, self.dC_buffer)])

    def update_status(self):
        # set the current client to idle
        self.cur_client.status = Status.IDLE

        # update the staleness
        if len(self.dW_buffer) >= self.buffer_size:
            for c in filter(lambda x: x.status == Status.ACTIVE, self.clients):
                self.staleness[c.id] += 1
            self.dW_buffer.clear()
            self.dC_buffer.clear()
            self.prev_model_buffer.clear()