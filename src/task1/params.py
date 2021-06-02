import os
from dataclasses import dataclass, asdict
from transformers import TrainingArguments

from ray import tune

from params import DefaultTrainingArguments, outputs_dir

@dataclass
class Task1Arguments(TrainingArguments):
    transformer: str = 'bert-base-cased'
    freeze_transformer: bool = False
    add_word_embs: bool = False
    add_amb_embs: bool = False

def get_model_name(args: Task1Arguments):
    name_parts = [args.transformer]
    if args.freeze_transformer:
        name_parts = ['frozen']
    if args.add_word_embs:
        name_parts += ['word-emb']
    if args.add_amb_embs:
        name_parts += ['amb-emb']
    name = '_'.join(name_parts)
    return name

def get_training_args_dict(cmd_args):
    training_args_dict = DefaultTrainingArguments().to_dict()
    training_args_dict.update(dict(
        output_dir = '',
        label_names = ["grade"],
        metric_for_best_model = 'rmse',
        report_to = 'comet_ml' if cmd_args.usecomet else "none",
        overwrite_output_dir = cmd_args.overwrite
    ))
    if cmd_args.num_epochs > 0:
        training_args_dict['num_train_epochs'] = cmd_args.num_epochs
    return training_args_dict


def get_args(cmd_args, search=False, **kwargs):
    args_dict = get_training_args_dict(cmd_args)
    if not search:
        args_dict.update(dict(
            transformer = cmd_args.transformer,
            freeze_transformer = cmd_args.freeze_transformer,
            add_word_embs = cmd_args.add_word_embs,
            add_amb_embs = cmd_args.add_amb_embs,
        ))
    args_dict.update(**kwargs)
    args = Task1Arguments(**args_dict)
    args.output_dir = os.path.join(outputs_dir, f'models/task1/{get_model_name(args)}')
    return args

def get_choice_fn(hyperopt=False):
    if hyperopt: return tune.choice
    else: return tune.grid_search

def get_hp_space(cmd_args):
    choices = get_choice_fn(hyperopt=cmd_args.usehyperopt)
    args = get_args(cmd_args=cmd_args, search=True, skip_memory_metrics=True)
    def hp_space_1(_):
        # hp_space = asdict(args)
        hp_space = {}
        hp_space.update(dict(
            transformer = choices(['bert-base-cased']),
            freeze_transformer = choices([True, False]),
            add_word_embs = choices([True, False]),
            add_amb_embs = choices([True, False])
        ))
        return hp_space
    hp_space = {
        '1': hp_space_1
    }[cmd_args.hpspace]
    is_grid = not cmd_args.usehyperopt
    return hp_space, is_grid
