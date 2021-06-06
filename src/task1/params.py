import comet_ml
import os
from dataclasses import dataclass, asdict
from transformers import TrainingArguments

from ray import tune

from params import get_training_args_dict, outputs_dir

@dataclass
class Task1Arguments(TrainingArguments):
    transformer: str = 'bert-base-cased'
    freeze_transformer: bool = True
    add_word_embs: bool = False
    add_amb_embs: bool = False
    add_amb_feat: bool = False

def get_model_name(args: Task1Arguments):
    name_parts = [args.transformer]
    if args.freeze_transformer:
        name_parts = ['frozen']
    if args.add_word_embs:
        name_parts += ['word-emb']
    if args.add_amb_embs:
        name_parts += ['amb-emb']
    if args.add_amb_feat:
        name_parts += ['amb-feat']
    name = '_'.join(name_parts)
    return name

def get_args(cmd_args, search=False, **kwargs):
    args_dict = get_training_args_dict(cmd_args)
    args_dict.update(dict(
        label_names = ["grade"],
        metric_for_best_model = 'eval_rmse',
        greater_is_better = False
    ))
    if not search:
        args_dict.update(dict(
            transformer = cmd_args.transformer,
            freeze_transformer = cmd_args.freeze_transformer,
            add_word_embs = cmd_args.add_word_embs,
            add_amb_embs = cmd_args.add_amb_embs,
            add_amb_feat = cmd_args.add_amb_feat,
        ))
    args_dict.update(**kwargs)
    args = Task1Arguments(**args_dict)
    args.output_dir = os.path.join(outputs_dir, f'models/task1/{get_model_name(args)}')
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
            freeze_transformer = choices([True, False]),
            add_word_embs = choices([True, False]),
            add_amb_embs = choices([True, False]),
            add_amb_feat = choices([True, False])
        ))
        return hp_space
    hp_space = {
        '1': hp_space_1
    }[cmd_args.hpspace]
    is_grid = not cmd_args.hyperopt
    return hp_space, is_grid
