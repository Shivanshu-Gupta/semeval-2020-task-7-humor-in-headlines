import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['COMET_PROJECT_NAME'] = 'humor-1'
os.environ['COMET_MODE'] = 'ONLINE'

import argparse
import pandas as pd
import comet_ml
import torchtext
from pprint import pprint as print
from pdb import set_trace
from transformers import AutoTokenizer, Trainer, TrainingArguments

from data import get_task1_dataset
from metrics import compute_metrics_task1 as compute_metrics
from params import models_dir, TrainingParams

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--overwrite", action='store_true')
parser.add_argument("--silent", action='store_true')
parser.add_argument("--usecomet", action='store_true')
parser.add_argument("--transformer", type=str, default='bert-base-cased')
parser.add_argument("--model_version", type=str, default='v1')
parser.add_argument("--num_epochs", type=int, default=0)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.transformer)
glove = None
if args.model_version in ['v2']:
    glove = torchtext.vocab.GloVe(name='840B', dim=300)
ds = get_task1_dataset(tokenizer=tokenizer, glove=glove, hl_w_mod=False)

for split in ds:
    print(f'{split}: {ds[split].shape}')
print(ds['train'].features)

if args.model_version == 'v1':
    from models import RegressionModelv1
    model = RegressionModelv1(args.transformer)
elif args.model_version == 'v2':
    from models import RegressionModelv2
    model = RegressionModelv2(transformer=args.transformer, word_emb_dim=glove.dim)

default_training_args = TrainingParams()
if args.num_epochs > 0:
    default_training_args.num_train_epochs = args.num_epochs

training_args = TrainingArguments(
    output_dir=os.path.join(models_dir, 'task1/v1'),
    label_names=["grade"],
    metric_for_best_model='rmse',
    report_to="comet_ml" if args.usecomet else "none",
    **default_training_args.to_dict()
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds['train'],
    eval_dataset=ds['validation'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()

