import argparse
import importlib
import yaml


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, help='Algorithm')

    # ===== Basic Setting ======
    parser.add_argument('--suffix', type=str, help="Suffix for file")
    parser.add_argument('--device', type=int, help="Device to use")
    parser.add_argument('--dataset', type=str, help="Dataset")
    parser.add_argument('--model', type=str, help="Model")

    # ===== Federated Setting =====
    parser.add_argument('--total_num', type=int, help="Total clients num")
    parser.add_argument('--sr', type=float, help="Clients sample rate")
    parser.add_argument('--rnd', type=int, help="Communication rounds")
    parser.add_argument('--test_gap', type=int, help='Rounds between test phases')

    # ===== Local Training Setting =====
    parser.add_argument('--bs', type=int, help="Batch size")
    parser.add_argument('--epoch', type=int, help="Epoch num")
    parser.add_argument('--lr', type=float, help="Learning rate")
    parser.add_argument('--gamma', type=float, help="Exponential decay of learning rate")

    # ===== Async Setting =====
    parser.add_argument('--decay', type=float, default=0.3, help="Basic weight decay in asynchronous aggregation")

    # ===== System Heterogeneity Setting =====
    parser.add_argument('--delay', type=int, default=5, help="Delay level used to simulate latency of device")
    parser.add_argument('--delay_rate', type=float, default=0.3, help="Proportion of stale device")


    # === read specific parameters from each method
    args, _ = parser.parse_known_args()
    alg_module = importlib.import_module(f'alg.{args.alg}')
    spec_args = alg_module.add_args(parser) if hasattr(alg_module, 'add_args') else args

    # === read params from yaml ===
    # NOTE: Only overwrite when the value is None
    with open('config.yaml', 'r') as f:
        yaml_config = yaml.load(f.read(), Loader=yaml.Loader)
    for k, v in vars(spec_args).items():
        if v is None: setattr(spec_args, k, yaml_config[k])
    return spec_args