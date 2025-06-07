from alg.asyncbase import AsyncBaseClient, AsyncBaseServer, Status
from utils.time_utils import time_record


def add_args(parser):
    parser.add_argument('--a', type=int, default=1)
    parser.add_argument('--b', type=int, default=4)
    parser.add_argument('--strategy', type=str, default='hinge', help='constant/poly/hinge')
    return parser.parse_args()


class Client(AsyncBaseClient):
    @time_record
    def run(self):
        self.train()


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
        def weight_decay():
            a = self.args.a
            b = self.args.b
            strategy = self.args.strategy
            tau = self.staleness[self.cur_client.id]
            if strategy == 'poly':
                return 1 / ((tau + 1) ** abs(a))
            elif strategy == 'hinge':
                return 1 / (a * (tau + b) + 1) if tau > b else 1
            else:
                return 1

        decay = self.decay * weight_decay()
        t_aggr = decay * self.cur_client.model2tensor() + (1 - decay) * self.model2tensor()
        self.tensor2model(t_aggr)