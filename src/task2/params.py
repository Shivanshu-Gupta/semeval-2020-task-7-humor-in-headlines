import os
from dataclasses import dataclass, asdict
from transformers import TrainingArguments

from ray import tune

from params import get_training_args_dict, outputs_dir

@dataclass
class Task2Arguments(TrainingArguments):
    transformer: str = 'bert-base-cased'
    combined: bool = False

def get_model_name(args: Task2Arguments):
    name_parts = [args.transformer]
    name = '_'.join(name_parts)
    return name

def get_args(cmd_args, search=False, **kwargs):
    args_dict = get_training_args_dict(cmd_args)
    args_dict.update(dict(
        metric_for_best_model = 'eval_accuracy',
        greater_is_better = True
    ))
    if not search:
        args_dict.update(dict(
            transformer = cmd_args.transformer,
            combined = cmd_args.combined
        ))
    args_dict.update(**kwargs)
    args = Task2Arguments(**args_dict)
    args.output_dir = os.path.join(outputs_dir, f'models/task2/{get_model_name(args)}')
    return args

def get_choice_fn(hyperopt=False):
    if hyperopt: return tune.choice
    else: return tune.grid_search

def get_hp_space(cmd_args):
    choices = get_choice_fn(hyperopt=cmd_args.hyperopt)
    args = get_args(cmd_args=cmd_args, search=True, skip_memory_metrics=True)
    def hp_space_1(_):
        # hp_space = asdict(args)
        hp_space = {}
        hp_space.update(dict(
            transformer = choices(['bert-base-cased']),
        ))
        return hp_space
    hp_space = {
        '1': hp_space_1
    }[cmd_args.hpspace]
    is_grid = not cmd_args.hyperopt
    return hp_space, is_grid
