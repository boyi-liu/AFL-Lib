# AFL-Lib

🔥 **Asynchronous Federated Learning (AFL)** Library and Benchmark.

Currently, we support: 

+ 14 federated learning baselines, including 4 synchronous baselines, and 12 asynchronous baselines.
+ Comprehensive simulation of system heterogeneity in asynchronous FL, including:
  + Device heterogeneity
  + Communication heterogeneity
  + Device dropout
+ 7 datasets covering modalities like image, text, sensor.
+ 3 data heterogeneity setup. 



## Supported Methods

#### Synchronous baselines

+ **FedAvg**, [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf), *AISTATS 2017*
+ **FedProx**, [Federated Optimization in Heterogeneous Networks](https://proceedings.mlsys.org/paper_files/paper/2020/file/1f5fe83998a09396ebe6477d9475ba0c-Paper.pdf), *MLSys 2020*
+ **MOON**, [Model Contrastive Federated Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Model-Contrastive_Federated_Learning_CVPR_2021_paper.pdf), *CVPR 2021*
+ **rFedAvg**, [Distribution-Regularized Federated Learning on Non-IID Data](https://ieeexplore.ieee.org/document/10184650), *ICDE 2023*

#### Asynchronous baselines

+ **FedAsync**, [Asynchronous Federated Optimization](https://arxiv.org/abs/1903.03934), _2019_
+ **ASO-Fed**, [Asynchronous Online Federated Learning for Edge Devices with Non-IID Data](https://ieeexplore.ieee.org/document/9378161), *BigData 2020*
+ **FedBuff**, [Federated Learning with Buffered Asynchronous Aggregation](https://proceedings.mlr.press/v151/nguyen22b/nguyen22b.pdf), _AISTATS 2022_
+ **PORT**, [How Asynchronous Could Federated Learning Be?](https://ieeexplore.ieee.org/document/9812885/), *IWQoS 2022*
+ **Pisces**, [Pisces: Efficient Federated Learning via Guided Asynchronous Training](https://dl.acm.org/doi/abs/10.1145/3542929.3563463), *SoCC 2022*
+ **Async-Drop**, [Efficient and Light-Weight Federated Learning via Asynchronous Distributed Dropout](https://proceedings.mlr.press/v206/dun23a/dun23a.pdf), *AISTATS 2023* 
+ **FedAC**, [Effcient Asynchronous Federated Learning with Prospective Momentum Aggregation and Fine-Grained Correction](https://ojs.aaai.org/index.php/AAAI/article/view/29603), _AAAI 2024_ 
+ **DAAFL**, [Data Disparity and Temporal Unavailability Aware Asynchronous Federated Learning for Predictive Maintenance on Transportation Fleets](https://ojs.aaai.org/index.php/AAAI/article/view/29467), *AAAI 2024*
+ **FADAS**, [FADAS: Towards Federated Adaptive Asynchronous Optimization](https://icml.cc/virtual/2024/poster/33327), *ICML 2024*
+ **CA2FL**, [Tackling the Data Heterogeneity in Asynchronous Federated Learning with Cached Update Calibration](https://iclr.cc/virtual/2024/poster/19456), *ICLR 2024*



## Guidance

### Step 1: Prepare your data

You only need to tune `dataset/config.yaml` to modify the config.

Then you can run dataset-specific file to generate dataset.

```bash
python generate_mnist.py
```



### Step 2: Implement your algorithm

#### Step 2.1 Create your file

Create a new file `{NAME}.py` inside the path `alg`.



#### Step 2.2 Extend the basic Client and Server

If you are working on a **synchronous** FL algorithm, just extend the Client and Server class in `alg/base.py`:

```python
from alg.base import BaseServer, BaseClient

class Client(BaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)

class Server(BaseServer):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)
```

Otherwise, if you are working on an **asynchronous** FL algorithm, you may extend the Client and Server in `alg.asyncbase.py`:

```python
from alg.asyncbase import AsyncBaseServer, AsyncBaseClient

class Client(AsyncBaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)

class Server(AsyncBaseServer):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)
```



#### Step 2.3 Config your algorithm-specific hyperparameters

And all **general** args could be found in `utils/options.py`.

For **algorithm-specific** hyperparameters, it is recommended to add a `add_args()` function inside your file.

```python
def add_args(parser):
    parser.add_argument('--{your_param}', type=int, default=1)
    return parser.parse_args()
```



#### Step 2.4 Implement your algorithms

> ‼️ We claim that each algorithm should **overwrite the function** `run()`, because it stands for the main workflow of your algorithm.
>
> 💡***You can overwrite or add any function as you want then!***

For **synchronous** FL baselines, the `run()` follows a basic pipeline of:

```python
from alg.base import BaseServer, BaseClient
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
```

For asynchronous FL baselines, the `run()` follows a basic pipeline of:

```python
from alg.asyncbase import AsyncBaseServer, AsyncBaseClient
from utils.time_utils import time_record

class Client(AsyncBaseClient):
    @time_record
    def run(self):
        self.train()


class Server(AsyncBaseServer):
    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()
        self.update_status()
```

>NOTE:  `@time_record` is a decorator which is defined in `utils/time_utils.py` to measure the training time.



### Step 3: Run your code!

Now it is time to run your code!

#### Step 3.1: Config your hyperparamters

There are three places to config your hyperparameters:

+ ⭐️⭐️⭐️ Highest priority: Your bash to run the code, for example, `python main.py --total_num 10`
+ ⭐️⭐️ Medium priority: The content in `config.yaml`
+ ⭐️ Lowest priority: The default setup in `utils/options.py`

>  The priority means that, if you change `total_num` to `10` in your bash, it will **overwrite** that in `config.yaml` and `utils/options.py`.



#### Step 3.2 Run your code!

```bash
python main.py --{your args1} {your args1's value} --{your args2} {your args2's value} ... --{your args-n} {your args-n's value}
```
