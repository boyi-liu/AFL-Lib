import torch
import numpy as np
from alg.asyncbase import AsyncBaseClient, AsyncBaseServer, Status
from utils.time_utils import time_record


def add_args(parser):
    parser.add_argument('--beta', type=float, default=0.9)
    parser.add_argument('--buffer_size', type=int, default=5)
    parser.add_argument('--eta_g', type=float, default=0.01)
    parser.add_argument('--eta_l', type=float, default=0.01)
    return parser.parse_args()


class Client(AsyncBaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.eta_l = args.eta_l
        self.prev_model = None
        self.grad_momentum = None

    def save_model(self):
        self.prev_model = self.model2tensor().clone().detach()

    def compute_delta(self):
        current_model = self.model2tensor()
        delta = current_model - self.prev_model

        if self.grad_momentum is None:
            self.grad_momentum = torch.zeros_like(delta)

        self.grad_momentum = 0.9 * self.grad_momentum + 0.1 * delta
        corrected_delta = delta + self.grad_momentum

        self.save_model()
        return corrected_delta

    @time_record
    def run(self):
        if self.prev_model is None:
            self.save_model()

        self.train()

        self.dW = self.compute_delta()


class Server(AsyncBaseServer):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)

        self.m_t = torch.zeros_like(self.model2tensor())
        self.beta = args.beta
        self.eta_g = args.eta_g

        self.buffer_size = args.buffer_size
        self.model_buffer = []
        self.update_buffer = []
        self.weight_buffer = []

        self.prev_model = None
        self.prev_loss = float('inf')
        self.lr_momentum = 0.0
        self.lr_beta = 0.9
        self.min_lr = 0.001
        self.max_lr = 0.1

    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()
        self.update_status()

    def calculate_weight(self, model_update):
        if model_update is None:
            return 0.0

        update_norm = torch.norm(model_update)
        global_norm = torch.norm(self.model2tensor())
        relative_update = update_norm / (global_norm + 1e-8)

        staleness = self.staleness[self.cur_client.id]
        staleness_factor = 1.0 / (1.0 + 0.05 * staleness)

        base_weight = torch.log1p(torch.exp(-relative_update))
        weight = base_weight * staleness_factor

        weight = max(weight.item(), 1e-3)
        return weight

    def update_learning_rate(self, current_model, current_loss):
        if self.prev_model is None:
            self.prev_model = current_model.clone()
            self.prev_loss = current_loss
            return

        loss_diff = self.prev_loss - current_loss

        if loss_diff > 0:
            progress = loss_diff / (self.prev_loss + 1e-8)
            lr_delta = 0.05 * progress
        else:
            regress = -loss_diff / (self.prev_loss + 1e-8)
            lr_delta = -0.1 * regress

        self.lr_momentum = self.lr_beta * self.lr_momentum + (1 - self.lr_beta) * lr_delta

        self.eta_g *= (1.0 + self.lr_momentum)
        self.eta_g = self.min_lr + (self.max_lr - self.min_lr) * \
                     (1 / (1 + np.exp(-(self.eta_g - (self.min_lr + self.max_lr) / 2))))

        self.prev_model = current_model.clone()
        self.prev_loss = current_loss

    def aggregate(self):
        client_model = self.cur_client.model2tensor()
        model_update = self.cur_client.dW
        if model_update is not None:
            model_update = model_update

        weight = self.calculate_weight(model_update)

        self.model_buffer.append(client_model)
        self.update_buffer.append(model_update if model_update is not None else torch.zeros_like(client_model))
        self.weight_buffer.append(weight)

        if len(self.model_buffer) >= self.buffer_size:
            try:
                weights = torch.tensor(self.weight_buffer, device=self.device)
                weights = weights / (weights.sum() + 1e-8)

                models = torch.stack(self.model_buffer)
                updates = torch.stack(self.update_buffer)

                avg_model = torch.sum(models * weights.view(-1, 1), dim=0)
                avg_update = torch.sum(updates * weights.view(-1, 1), dim=0)

                self.m_t = self.beta * self.m_t + (1 - self.beta) * avg_update

                x_final = avg_model + self.eta_g * self.m_t

                current_loss = torch.norm(self.m_t).item()

                self.update_learning_rate(avg_model, current_loss)
                self.tensor2model(x_final)

            except Exception as e:
                print(f"Aggregation error: {str(e)}")

            self.model_buffer = []
            self.update_buffer = []
            self.weight_buffer = []