import re
import comet_ml
from sklearn import metrics
import torch
import datasets
import transformers
import pandas as pd
from datasets import load_dataset, ClassLabel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
comet_ml.init(project_name='humor-2')

experiment = Experiment(project_name="basic humor classification")

# ds1 = load_dataset("humicroedit", "subtask-1")
ds: datasets.DatasetDict = load_dataset("humicroedit", "subtask-2")
label_names = ds['train'].features['label'].names
model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=3)
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

def make_headlines_with_mod(ex):
    def make_headline(s, e):
        p = re.compile(r'<(.*)/>')
        return p.sub(f'[\\1|{e}]', s)
    h1 = make_headline(ex['original1'], ex['edit1'])
    h2 = make_headline(ex['original2'], ex['edit2'])
    return {'headline1': h1, 'headline2': h2, 'combined': f"{h1} [SEP] {h2}"}


def tokenize(examples):
    encoded = tokenizer(examples['original1'], examples['original2'])
    return tokenizer.decode(encoded['input_ids'])


def encode(examples):
    return tokenizer(examples['original1'], examples['original2'], truncation=True, padding='max_length')


ds = ds.map(make_headlines_with_mod)
binary_ds = ds.filter(lambda ex: ex['label'] != 0).\
    map(lambda ex: {'label': ex['label'] - 1})
binary_ds_features = ds['train'].features.copy()
binary_ds_features['label'] = ClassLabel(names=ds['train'].features['label'].names[1:])
binary_ds = binary_ds.cast(binary_ds_features)
encoded_ds = binary_ds.map(encode, batched=True).\
    map(lambda examples: {'labels': examples['label']}, batched=True)
torch_ds = encoded_ds.copy()
for split in torch_ds:
    torch_ds[split].set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

use_gpu = True
gpu_idx = 1
device = f'cuda:{gpu_idx}' if torch.cuda.is_available() and use_gpu else 'cpu'
model.train().to(device)

EPOCHS = 5
WEIGHT_DECAY = 0.99


def get_example(index):
    return ds['validation'][index]['combined']


def compute_metrics(pred):
    experiment = comet_ml.get_global_experiment()

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)

    if experiment:
        epoch = int(experiment.curr_epoch) if experiment.curr_epoch is not None else 0
        experiment.set_epoch(epoch)
        experiment.log_confusion_matrix(
            y_true=labels,
            y_predicted=preds,
            file_name=f"confusion-matrix-epoch-{epoch}.json",
            labels=label_names,
            index_to_example_function=get_example
        )

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


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
