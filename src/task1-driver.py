import re
import comet_ml
from sklearn import metrics
import torch
from torch import nn
import datasets
import transformers
import pandas as pd
from datasets import load_dataset, ClassLabel
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import os

comet_ml.init(project_name='humor-1')

ds: datasets.DatasetDict = load_dataset("humicroedit", "subtask-1")
bert_model = AutoModel.from_pretrained('bert-base-cased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

def make_headlines(sample):
    line = sample["original"]
    edit = sample["edit"]

    opening = line.index("<")
    closing = line.index("/>", opening)

    start = line[:opening]
    end = line[closing + 2:]
    original_word = line[opening+1:closing]

    original = start + original_word + end
    edited = start + edit + end
    
    return {
        'original_sentence': original,
        'edited_sentence': edited,
        'original_word': original_word, 
        'edited_word': edit, 
        'grade': sample['meanGrade']
    }

def encode(examples):
    return tokenizer(
        examples['original_sentence'], 
        examples['edited_sentence'], 
        truncation=True, 
        padding='max_length'
    )


ds = ds.map(make_headlines)
encoded_ds = ds.map(encode, batched=True)
torch_ds = encoded_ds.copy()
for split in torch_ds:
    torch_ds[split].set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'grade'])

use_gpu = True
gpu_idx = 1
device = f'cuda:{gpu_idx}' if torch.cuda.is_available() and use_gpu else 'cpu'

EPOCHS = 5
WEIGHT_DECAY = 0.99

class RegressionModel(nn.Module):
    def __init__(self, bert_model):
        super(RegressionModel, self).__init__()
        
        self.bert = bert_model.eval()

        self.l1 = nn.Linear(768, 256)
        self.l2 = nn.Linear(256, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            bert_out = bert_model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids
            ).pooler_output
            
        x = torch.tanh(self.l1(bert_out))
        x = self.l2(x)
        return x

# def get_example(index):
#     return ds['validation'][index]['combined']


# def compute_metrics(pred):
#     experiment = comet_ml.get_global_experiment()

#     labels = pred.label_ids
#     preds = pred.predictions.argmax(-1)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
#     acc = accuracy_score(labels, preds)

#     if experiment:
#         epoch = int(experiment.curr_epoch) if experiment.curr_epoch is not None else 0
#         experiment.set_epoch(epoch)
#         experiment.log_confusion_matrix(
#             y_true=labels,
#             y_predicted=preds,
#             file_name=f"confusion-matrix-epoch-{epoch}.json",
#             labels=label_names,
#             index_to_example_function=get_example
#         )

#     return {
#         'accuracy': acc,
#         'f1': f1,
#         'precision': precision,
#         'recall': recall
#     }

model = RegressionModel(bert_model).to(device)

# The commented code works
# train_dl = DataLoader(torch_ds['train'], batch_size=32)

# sample = next(iter(train_dl)) 
# for key in sample:
#     sample[key] = sample[key].to(device)

# model(sample['input_ids'], sample['attention_mask'], sample['token_type_ids'])

training_args = TrainingArguments(
    seed=42,
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.9,
    learning_rate=1e-5,
    evaluation_strategy="epoch",
    do_train=True,
    do_eval=True
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=torch_ds['train'],
    eval_dataset=torch_ds['validation'],
    compute_metrics=compute_metrics,
)
trainer.train()

