import argparse
from typing import Union
from pprint import pprint as print
from datasets.dataset_dict import DatasetDict
import torch

def output(string='\n', filepath=None, silent=False):
    if not silent: print(string)
    if filepath is not None:
        with open(filepath, 'w') as outf:
            outf.write(string + '\n')

def print_ds_stats(ds: DatasetDict, **kwargs):
    for split in ds:
        output(f'{split}: {ds[split].shape}', **kwargs)
    output(ds['train'].features, **kwargs)

def get_common_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--overwrite', action='store_true')
    parser.add_argument('--silent', action='store_true')
    parser.add_argument('--comet', action='store_true')
    parser.add_argument('--num_epochs', type=int, default=0)
    return parser

def create_object_from_class_string(module_name, class_name, parameters):
    import importlib
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    instance = class_(**parameters)
    return instance

def load_object_from_dict(parameters, **kwargs):
    if parameters is None:
        return None
    if not isinstance(parameters, dict):
        parameters = vars(parameters).copy()
    parameters.update(kwargs)
    type = parameters.pop('type')
    if type:
        type = type.split('.')
        module_name, class_name = '.'.join(type[:-1]), type[-1]
        return create_object_from_class_string(module_name, class_name, parameters)

def get_device(gpu_idx):
    device = 'cpu'
    if gpu_idx != -1 and torch.cuda.is_available():
        device = f'cuda:{gpu_idx}'
    return device
