import importlib

dataset2class = {
    'mnist': 10,
    'emnist': 47,
    'cifar10': 10,
    'cifar100': 100,
    'svhn': 10,
    'agnews': 4,
    'shakespeare': 4,
    'harbox': 5
}

# NOTE: if input is 'cifar10a1', then process into 'cifar10'
def name_filter(dataset_arg):
    if 'cifar100' in dataset_arg:
        return 'cifar100'
    elif 'cifar10' in dataset_arg:
        return 'cifar10'
    elif 'harbox' in dataset_arg:
        return 'harbox'
    elif 'agnews' in dataset_arg:
        return 'agnews'
    elif 'mnist' in dataset_arg:
        return 'mnist'
    return 'Unknown dataset'


def load_model(args):
    dataset_arg = name_filter(args.dataset)
    args.class_num = dataset2class[dataset_arg]
    if args.class_num == -1:
        exit('Dataset params not exist (in config.py)!')

    model_arg = args.model
    model_module = importlib.import_module(f'model.{model_arg}')
    return getattr(model_module, f'{model_arg}_{dataset_arg}')(args)