import json
import attr
from typing import Union, Optional

from param_impl import Parameters

outputs_dir = '../outputs/'

def disambiguate(o, t):
    lambdas = {
        Union[int, str]: lambda *_: None
    }
    if t in lambdas:
        return lambdas[t](o, t)
    else:
        raise TypeError("Unknown Type")

@attr.s(auto_attribs=True)
class DefaultTrainingArguments(Parameters):
    output_dir: str = ''
    seed: int = 42
    overwrite_output_dir: bool = True
    num_train_epochs:int = 5
    per_device_train_batch_size:int = 128
    per_device_eval_batch_size:int = 512
    remove_unused_columns: bool = False
    warmup_steps:int = 500
    weight_decay: float = 0.9
    learning_rate: float = 5e-5
    evaluation_strategy: str = "epoch"
    logging_strategy: str = "epoch"
    do_train: bool = True
    do_eval: bool = True
    load_best_model_at_end: bool = True

@attr.s(auto_attribs=True)
class MetricParams(Parameters):
    name: str = 'accuracy'
    direction: str = 'maximize'
    mode: str = 'max'
    def __attrs_post_init__(self):
        self.mode = self.direction[:3]

def get_training_args_dict(cmd_args):
    training_args_dict = DefaultTrainingArguments().to_dict()
    training_args_dict.update(dict(
        report_to = 'comet_ml' if cmd_args.comet else "none",
        overwrite_output_dir = cmd_args.overwrite
    ))
    if cmd_args.num_epochs > 0:
        training_args_dict['num_train_epochs'] = cmd_args.num_epochs
    return training_args_dict
