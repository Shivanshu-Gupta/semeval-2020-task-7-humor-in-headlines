import os
import json
import attr
from typing import Union, Optional

from param_impl import default_value, Parameters

paths = json.load(open('src/paths.json'))
data_dir = paths['data_dir']
embeddings_dir = paths['embeddings_dir']
log_dir = paths['log_dir']
models_dir = paths['models_dir']
ray_dir = paths['ray_dir']

def disambiguate(o, t):
    lambdas = {
        Union[int, str]: lambda *_: None
    }
    if t in lambdas:
        return lambdas[t](o, t)
    else:
        raise TypeError("Unknown Type")

@attr.s(auto_attribs=True)
class TrainingParams(Parameters):
    seed: int = 42
    overwrite_output_dir: bool = True
    num_train_epochs:int = 50
    per_device_train_batch_size:int = 128
    per_device_eval_batch_size:int = 512
    remove_unused_columns: bool = False
    warmup_steps:int = 200
    learning_rate: float = 5e-4
    evaluation_strategy: str = "epoch"
    logging_strategy: str = "epoch"
    do_train: bool = True
    do_eval: bool = True
    load_best_model_at_end: bool = True
