from argparse import ArgumentParser
import comet_ml
import os
from dataclasses import dataclass, asdict
from transformers import TrainingArguments

from ray import tune

from params import TaskArguments, get_training_args_dict, outputs_dir

@dataclass
class Task1Arguments(TaskArguments):
    transformer: str = 'bert-base-cased'    # bert-{base, large}-cased, roberta-{base, large}, distilbert-base-cased, distilroberta-base
    freeze_transformer: bool = True
    add_word_embs: bool = False
    add_amb_embs: bool = False
    add_amb_feat: bool = False

    def model_name(self):
        name_parts = [self.transformer]
        if self.freeze_transformer:
            name_parts += ['frozen']
        if self.add_word_embs:
            name_parts += ['word-emb']
        if self.add_amb_embs:
            name_parts += ['amb-emb']
        if self.add_amb_feat:
            name_parts += ['amb-feat']
        name = '_'.join(name_parts)
        return name

def setup_parser(parser: ArgumentParser):
    Task1Arguments.setup_parser(parser)

def get_args(cmd_args, search=False, **kwargs):
    args_dict = get_training_args_dict(cmd_args)
    args_dict.update(dict(
        label_names = ["grade"],
        metric_for_best_model = 'eval_rmse',
        greater_is_better = False
    ))
    if not search:
        args_dict.update(Task1Arguments.parse_args(cmd_args=cmd_args))
    args_dict.update(**kwargs)
    args = Task1Arguments(**args_dict)
    args.output_dir = os.path.join(outputs_dir, f'models/task-1/{args.model_name()}')
    return args

def get_choice_fn(hyperopt=False):
    if hyperopt: return tune.choice
    else: return tune.grid_search

def get_hp_space(cmd_args):
    choices = get_choice_fn(hyperopt=cmd_args.hyperopt)
    hp_spaces = {
        'base0': dict(transformer=choices(['bert-base-cased', 'roberta-base', 'distilbert-base-cased', 'distilroberta-base'])),
        'base1': dict(
            transformer=choices(['bert-base-cased', 'roberta-base', 'distilbert-base-cased', 'distilroberta-base']),
            freeze_transformer=False
            ),
        'base2': dict(
            transformer=choices(['bert-base-cased', 'roberta-base', 'distilbert-base-cased', 'distilroberta-base']),
            freeze_transformer = choices([True, False]),
            add_word_embs = choices([True, False]),
            learning_rate=choices([1e-5, 3e-5, 1e-4, 3e-4, 1e-3]),
            weight_decay=choices([0.05, 0.1, 0.2, 0.4, 0.8]),
            add_amb_embs = False
            ),
        'amb_embs': dict(
            transformer=choices(['bert-base-cased', 'roberta-base', 'distilbert-base-cased', 'distilroberta-base']),
            freeze_transformer = choices([True, False]),
            add_word_embs = choices([True, False]),
            add_amb_embs = choices([True, False]),
            add_amb_feat = choices([True, False])
            )
    }
    hp_space = lambda _: hp_spaces[cmd_args.hp_space]
    is_grid = not cmd_args.hyperopt
    return hp_space, is_grid
