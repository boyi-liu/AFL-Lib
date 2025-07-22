import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from alg.clusterbase import ClusterClient, ClusterServer, Cluster
from utils.time_utils import time_record


def add_args(parser):
    parser.add_argument('--eps_1', type=float, default=0.4)
    parser.add_argument('--eps_2', type=float, default=0.7)
    return parser.parse_args()


class Client(ClusterClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.dW = self.model2tensor()

    @time_record
    def run(self):
        t_old = self.model2tensor()
        self.train()
        self.dW = self.model2tensor() - t_old

class Server(ClusterServer):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)
        self.eps_1 = args.eps_1
        self.eps_2 = args.eps_2
        self.sims = np.zeros([self.client_num, self.client_num])

    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()
        self.cluster()

    def cluster(self):
        self.update_sims()
        for cluster_id, cluster in enumerate(self.cluster_list):
            clients = [c for c in self.clients if c.cluster_id == cluster_id]
            client_ids = [c.id for c in clients]

            max_norm = max_update_norm(clients)
            mean_norm = mean_update_norm(clients)


            if mean_norm < self.eps_1 and max_norm > self.eps_2 and len(clients) > 2:
                similarity_in_cluster = self.sims[client_ids][:, client_ids]
                cluster_res = AgglomerativeClustering(n_clusters=2,
                                                      affinity="precomputed",
                                                      linkage="complete").fit(-similarity_in_cluster)

                # c1 is kept for the original cluster
                c1 = np.argwhere(cluster_res.labels_ == 0).flatten()
                # c2 is applied for the new cluster
                c2 = np.argwhere(cluster_res.labels_ == 1).flatten()

                for c_id in c2:
                    self.clients[c_id].cluster_id = len(self.cluster_list)
                self.cluster_list.append(
                    Cluster(len(self.cluster_list), cluster.model)
                )


    def update_sims(self):
        for c_i in self.sampled_clients:
            for c_j in self.clients:
                idx_i = c_i.id
                idx_j = c_j.id
                self.sims[idx_i, idx_j] = self.sims[idx_j, idx_i] = torch.nn.functional.cosine_similarity(
                    self.clients[idx_i].dW,
                    self.clients[idx_j].dW,
                    dim=0)


def max_update_norm(clients):
    return np.max([torch.norm(client.dW).item() for client in clients])

def mean_update_norm(clients):
    return torch.norm(torch.mean(torch.stack([client.dW for client in clients]), dim=0)).item()