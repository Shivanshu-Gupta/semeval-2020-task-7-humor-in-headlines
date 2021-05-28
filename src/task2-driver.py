import os
# os.environ["COMET_LOGGING_FILE"] = "comet.log"
# os.environ["COMET_LOGGING_FILE_LEVEL"] = "debug"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['COMET_PROJECT_NAME'] = 'humor-2'
os.environ['COMET_MODE'] = 'ONLINE'

import argparse
import pandas as pd
import comet_ml
from pprint import pprint as print

import torch
from datasets.dataset_dict import DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, EvalPrediction

from data import get_task2_dataset
from metrics import get_compute_metrics_task2
from params import models_dir, TrainingParams

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--overwrite", action='store_true')
parser.add_argument("--silent", action='store_true')
parser.add_argument("--usecomet", action='store_true')
parser.add_argument("--transformer", type=str, default='bert-base-cased')
parser.add_argument("--num_epochs", type=int, default=0)
args = parser.parse_args()

model = AutoModelForSequenceClassification.from_pretrained(args.transformer, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(args.transformer)
ds: DatasetDict = get_task2_dataset(tokenizer=tokenizer, combined=True)
for split in ds:
    print(f'{split}: {ds[split].shape}')
print(ds['train'].features)
label_names = ds['train'].features['labels'].names
compute_metrics = get_compute_metrics_task2(tokenizer=tokenizer, ds=ds, label_names=label_names)

default_training_args = TrainingParams()
if args.num_epochs > 0:
    default_training_args.num_train_epochs = args.num_epochs

training_args = TrainingArguments(
    seed=42,
    output_dir=os.path.join(models_dir, 'task2/v1'),
    metric_for_best_model='accuracy',
    report_to="comet_ml" if args.usecomet else "none",
    **default_training_args.to_dict()
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds['train'],
    eval_dataset=ds['validation'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()
