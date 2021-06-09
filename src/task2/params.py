import os
from argparse import ArgumentParser
from dataclasses import dataclass

from ray import tune

from params import TaskArguments, get_training_args_dict, outputs_dir

@dataclass
class Task2Model0Arguments(TaskArguments):
    model_id: int = 0   # asdasdasd
    checkpoint_path: str = None

    # Arguments for Task 1 Regression Model
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

@dataclass
class Task2Model1Arguments(TaskArguments):
    model_id: int = 1
    transformer: str = 'bert-base-cased'    # bert-{base, large}-cased, roberta-{base, large}, distilbert-base-cased, distilroberta-base
    def model_name(self):
        return f'model1_{self.transformer}'

@dataclass
class Task2Model2Arguments(TaskArguments):
    model_id: int = 2
    transformer: str = 't5-base'    # t5-{small, base, large}
    def model_name(self):
        return f'model2_{self.transformer}'

model_params_classes = [
    Task2Model0Arguments,
    Task2Model1Arguments,
    Task2Model2Arguments
]

def setup_parser(parser: ArgumentParser):
    subparsers = parser.add_subparsers(help='Whick Task 2 model?', dest='model')
    for idx in range(len(model_params_classes)):
        subparser = subparsers.add_parser(f'model-{idx}', help=f'Task 2 model {idx} arguments')
        model_params_classes[idx].setup_parser(subparser)

def get_args(cmd_args, model_id, search=False, **kwargs):
    args_dict = get_training_args_dict(cmd_args)
    args_dict.update(dict(
        metric_for_best_model = 'eval_accuracy',
        greater_is_better = True
    ))
    model_params_cls = model_params_classes[model_id]
    if not search:
        args_dict.update(model_params_cls.parse_args(cmd_args=cmd_args))
    args_dict.update(**kwargs)
    args = model_params_cls(**args_dict)
    args.output_dir = os.path.join(outputs_dir, f'models/task-2/{args.model_name()}')
    return args

def get_choice_fn(hyperopt=False):
    if hyperopt: return tune.choice
    else: return tune.grid_search

def get_hp_space(cmd_args):
    choices = get_choice_fn(hyperopt=cmd_args.hyperopt)
    # args = get_args(cmd_args=cmd_args, search=True, skip_memory_metrics=True)
    hp_spaces = {
        'model1_base': dict(
            transformer=choices(['bert-base-cased', 'roberta-base', 'distilbert-base-cased', 'distilroberta-base']),
            per_device_train_batch_size=choices([32, 64, 128]),
            learning_rate=choices([1e-6, 3e-5, 1e-4, 3e-3, 1e-2]),
            weight_decay=choices([0.8, 0.9])
            ),
        'model1_base2': dict(
            transformer=choices(['bert-base-cased', 'roberta-base', 'distilbert-base-cased', 'distilroberta-base']),
            per_device_train_batch_size=choices([32, 64, 128]),
            learning_rate=choices([1e-5, 3e-5, 1e-4, 3e-4, 1e-3]),
            weight_decay=choices([0.8, 0.9])
            ),
        # 'model1_0': dict(transformer = choices(['bert-base-cased', 'bert-large-cased'])),
        'model2_0': dict(transformers=choices(['t5-base', 'tf-large']))
    }
    hp_space = lambda _: hp_spaces[cmd_args.hp_space]
    is_grid = not cmd_args.hyperopt
    return hp_space, is_grid

# results:
# model1_base:
#     {
#         'transformer': 'bert-base-cased',
#         'per_device_train_batch_size': 32,
#         'learning_rate': 3e-05,
#         'weight_decay':0.8
#     }