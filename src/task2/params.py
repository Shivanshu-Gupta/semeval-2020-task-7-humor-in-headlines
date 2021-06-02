from dataclasses import dataclass
from transformers import TrainingArguments

from params import CommonDefaultParams

@dataclass
class Task2Arguments(TrainingArguments):
    transformer: str = 'bert-base-cased'

def get_args(**kwargs):
    args_dict = CommonDefaultParams().to_dict()
    args_dict.update(kwargs=kwargs)
    args = Task2Arguments(**args_dict)
    return args