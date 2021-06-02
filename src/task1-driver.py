import os

# Better to set CUDA_VISIBLEDEVICES while running the script as: `CUDA_VISIBLE_DEVICES=0 python task1-driver.py <options>``
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['COMET_PROJECT_NAME'] = 'humor-1'
os.environ['COMET_MODE'] = 'ONLINE'

import argparse
import pandas as pd
import comet_ml
from pprint import pprint as print
from pdb import set_trace
from transformers import AutoTokenizer, Trainer, TrainingArguments

from data import get_task1_dataset
from models import RegressionModel
from metrics import get_compute_metrics_task1
from params import models_dir, TrainingParams

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--overwrite", action='store_true')
parser.add_argument("--silent", action='store_true')
parser.add_argument("--usecomet", action='store_true')
parser.add_argument("--transformer", type=str, default='bert-base-cased')
parser.add_argument("--freeze", "--freeze_transformer", action="store_true")
parser.add_argument("--add_word_embs", action='store_true')
parser.add_argument("--add_amb_embs", action='store_true')
parser.add_argument("--num_epochs", type=int, default=0)
args = parser.parse_args()

print(args)

tokenizer = AutoTokenizer.from_pretrained(args.transformer)
ds = get_task1_dataset(tokenizer=tokenizer,
                       add_word_embs=args.add_word_embs,
                       add_amb_embs=args.add_amb_embs,
                       hl_w_mod=False)

for split in ds:
    print(f'{split}: {ds[split].shape}')
print(ds['train'].features)

word_emb_dim = ds['train'][0]['word_ini_emb'].shape[0] if args.add_word_embs else 0
amb_emb_dim = ds['train'][0]['amb_emb_ini'].shape[0] if args.add_amb_embs else 0

model = RegressionModel(transformer=args.transformer,
                        word_emb_dim=word_emb_dim,
                        amb_emb_dim=amb_emb_dim)
#print(model)
training_params = TrainingParams()
if args.num_epochs > 0:
    training_params.num_train_epochs = args.num_epochs

compute_metrics = get_compute_metrics_task1()

training_args: TrainingArguments = training_params.instantiate(
    output_dir=os.path.join(models_dir, f'task1/{model.name}'),
    label_names=["grade"],
    metric_for_best_model='rmse',
    report_to="comet_ml" if args.usecomet else "none",
)

trainer: Trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds['train'],
    eval_dataset=ds['validation'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()
trainer.evaluate()
