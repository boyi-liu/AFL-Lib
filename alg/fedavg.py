from alg.base import BaseClient, BaseServer
from utils.time_utils import time_record

class Client(BaseClient):
    @time_record
    def run(self):
        self.train()

class Server(BaseServer):
    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()