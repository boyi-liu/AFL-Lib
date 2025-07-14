import torch

from alg.asyncbase import AsyncBaseClient, AsyncBaseServer, Status
from utils.time_utils import time_record


def add_args(parser):
    parser.add_argument('--M', type=int, default=5, help='buffer size M for FADAS')
    parser.add_argument('--tau_c', type=int, default=1, help='delay threshold')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for AMSGrad')
    parser.add_argument('--beta2', type=float, default=0.99, help='beta2 for AMSGrad')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='epsilon for AMSGrad')
    parser.add_argument('--eta', type=float, default=0.001, help='global learning rate')
    
    return parser.parse_args()


class Client(AsyncBaseClient):
    @time_record
    def run(self):
        w_before = self.model2tensor()
        self.train()
        self.dW = self.model2tensor() - w_before


class Server(AsyncBaseServer):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)
        self.buffer = []
        self.buffer_delays = []
        self.m = None 
        self.v = None 
        self.v_hat = None  
        self.t = 0  
        
        self.M = args.M
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.epsilon = args.epsilon
        self.eta = args.eta
        self.tau_c = args.tau_c
        
        self.client_start_round = {}

    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()
        self.update_status()

    def aggregate(self):
        client_id = self.cur_client.id
        client_delay = self.staleness[client_id]
        
        self.buffer.append(self.cur_client.dW)
        self.buffer_delays.append(client_delay)
        
        if len(self.buffer) >= self.M:
            self._perform_global_update()

    def update_status(self):
        # set the current client to idle
        self.cur_client.status = Status.IDLE

        # update the staleness
        if len(self.buffer) >= self.M:
            for c in filter(lambda x: x.status == Status.ACTIVE, self.clients):
                self.staleness[c.id] += 1
            self.buffer.clear()
            self.buffer_delays.clear()

    def _perform_global_update(self):
        Delta_t = torch.mean(torch.stack(self.buffer), dim=0)
        
        if self.m is None:
            self.m = torch.zeros_like(Delta_t)
            self.v = torch.zeros_like(Delta_t)
            self.v_hat = torch.zeros_like(Delta_t)
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * Delta_t
        self.v = self.beta2 * self.v + (1 - self.beta2) * (Delta_t ** 2)
        self.v_hat = torch.maximum(self.v_hat, self.v)
        max_delay = max(self.buffer_delays)
        eta_t = min(self.eta, self.eta / max_delay) if max_delay > self.tau_c else self.eta

        t_g = self.model2tensor()
        t_g_new = t_g + eta_t * self.m / (torch.sqrt(self.v) + self.epsilon)
        self.tensor2model(t_g_new)