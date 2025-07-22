import torch

from alg.base import BaseClient, BaseServer


class ClusterClient(BaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.cluster_id = 0

    def clone_model(self, target):
        target_cluster = target.cluster_list[self.cluster_id]
        p_tensor = target_cluster.model
        self.tensor2model(p_tensor)


class ClusterServer(BaseServer):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)

        self.cluster_num = args.cluster_num if 'cluster_num' in args.__dict__ else 1
        assert self.cluster_num > 0
        self.cluster_list = [Cluster(idx, self.model2tensor()) for idx in range(self.cluster_num)]

    def uplink(self):
        assert (len(self.sampled_clients) > 0)
        def nan_to_zero(tensor):
            return torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
        for cluster in self.cluster_list:
            cluster.received_params = [nan_to_zero(client.model2tensor())
                                       for client in self.sampled_clients
                                       if client.cluster_id == cluster.id]

    def aggregate(self):
        assert (len(self.sampled_clients) > 0)
        for cluster in self.cluster_list:
            total_samples = sum(len(client.dataset_train) for client in self.sampled_clients if client.cluster_id == cluster.id)
            weights = [len(client.dataset_train) / total_samples for client in self.sampled_clients if client.cluster_id == cluster.id]

            cluster.received_params = [params * weight for weight, params in zip(weights, cluster.received_params)]
            if len(cluster.received_params) != 0:
                cluster.model = sum(cluster.received_params)


class Cluster:
    def __init__(self, id, model):
        self.id = id
        self.model = model
        self.received_params = []