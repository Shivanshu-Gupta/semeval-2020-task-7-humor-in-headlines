import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['COMET_PROJECT_NAME'] = 'humor-1'
os.environ['COMET_MODE'] = 'ONLINE'

import comet_ml
import re
import torch
from torch import nn
import torch.nn.functional as F
import datasets
import transformers
from datasets import load_dataset, ClassLabel
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from tqdm import tqdm


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
        'grade': [sample['meanGrade']]
    }


def add_word_embeddings(sample):
    words = [sample['original_word'], sample['edited_word']]
    word_embs = glove.get_vecs_by_tokens(words, lower_case_backup=True)
    original_word_emb, edited_word_emb = word_embs    
    return {
        'original_word_emb': original_word_emb.numpy(),
        'edited_word_emb': edited_word_emb.numpy()
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

# use_gpu = True
# gpu_idx = 1
# device = f'cuda:{gpu_idx}' if torch.cuda.is_available() and use_gpu else 'cpu'

EPOCHS = 5
WEIGHT_DECAY = 0.99

class RegressionModel(nn.Module):
    def __init__(self, sentence_embedder):
        super(RegressionModel, self).__init__()
        
        self.sentence_embedder = sentence_embedder.eval()

        self.l1 = nn.Linear(768, 256)
        self.l2 = nn.Linear(256, 256)
        self.lout = nn.Linear(256, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            sentence_emb = self.sentence_embedder(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids
            ).pooler_output
            
        x = F.relu(self.l1(sentence_emb))
        x = F.relu(self.l2(x))
        x = self.lout(x)
        return x


# def compute_metrics(EvalPrediction):
#     grades = torch.from_numpy(EvalPrediction.label_ids)
#     preds = torch.from_numpy(EvalPrediction.predictions)
#     rmse = torch.sqrt(F.mse_loss(preds, grades))

#     return { 'rmse': rmse }

model = RegressionModel(bert_model)

class RegressionTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(RegressionTrainer, self).__init__(*args, **kwargs)

        self.loss_fn = torch.nn.MSELoss()
        self.eps = 1e-6

    def compute_loss(self, model, inputs, return_outputs=False):
        y = inputs.pop("grade")
        y_pred = model(**inputs)
        loss = torch.sqrt(self.loss_fn(y_pred, y) + self.eps)
        return (loss, y) if return_outputs else loss

training_args = TrainingArguments(
    seed=42,
    output_dir='./results',
    label_names=["grade"],
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    remove_unused_columns=False,
    warmup_steps=500,
    weight_decay=0.9,
    learning_rate=1e-5,
    evaluation_strategy="epoch",
    do_train=True,
    do_eval=True
)
trainer = RegressionTrainer(
    model=model,
    args=training_args,
    train_dataset=torch_ds['train'],
    eval_dataset=torch_ds['validation'],
#     compute_metrics=compute_metrics,
)
trainer.train()

