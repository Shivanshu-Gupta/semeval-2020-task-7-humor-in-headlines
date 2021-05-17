import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchtext
import random
import pandas as pd
from collections import namedtuple
from os import path

TASK1_BASE_PATH = path.dirname(path.abspath(__file__))
TASK1_BASE_PATH = path.abspath(path.join(TASK1_BASE_PATH, "../data/task-1"))


class HumicroeditTask1Dataset(Dataset):

    def __init__(self, split="train", transform=None):
        if split == "train":
            self.data = pd.read_csv(path.join(TASK1_BASE_PATH, "train.csv"))
        elif split == "dev":
            self.data = pd.read_csv(path.join(TASK1_BASE_PATH, "dev.csv"))
        elif split == "test":
            self.data = pd.read_csv(path.join(TASK1_BASE_PATH, "test.csv"))
        
        self.HumicreoditItem = namedtuple(
            'HumicreoditItem', 
            ('original_sentence', 'edited_sentence', 'original_word', 'edited_word', 'grade'))
            
        self.transform = transform
        
    def _create_item_from_row(self, row):
        line = row["original"]
        edit = row["edit"]
        
        opening = line.index("<")
        closing = line.index("/>", opening)
        
        start = line[:opening]
        end = line[closing + 2:]
        original_word = line[opening+1:closing]
        
        original = start + original_word + end
        edited = start + edit + end
        
        return self.HumicreoditItem(
            original_sentence=original, 
            edited_sentence=edited,
            original_word=original_word,
            edited_word=edit,
            grade=row['meanGrade'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        items = self.data.iloc[idx]
        if len(items.shape) > 1:
            sample = items.apply(self._create_item_from_row, axis=1).tolist()
        else:
            sample = self._create_item_from_row(items)

        if self.transform:
            if isinstance(sample, list):
                sample = [self.transform(s) for s in sample]
            else:
                sample = self.transform(sample)

        return sample

device = "cuda:1" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
print(f"Using: {device}")

EPOCHS = 2
LEARNING_RATE = 1e-5
BATCH_SIZE = 24


bert_tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
bert_model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')


class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()

        self.l1 = nn.Linear(768, 256)
        self.l2 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = self.l2(x)
        return x


model = RegressionModel()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)

TrainingItem = namedtuple('TrainingItem', ('x', 'y'))

def transform(dataset_item):
    """
    Transforms Dataset tuples into tokens with BERT's tokenizer.
    Returns
        TrainingItem namedtuple with x=torch.tensor(tokens) and y=torch.tensor(grade)
    """
    indexed_tokens = bert_tokenizer.encode(
        dataset_item.edited_sentence,
        padding="max_length")
    
    tokens = torch.as_tensor(indexed_tokens, dtype=torch.long)
    grade = torch.as_tensor([dataset_item.grade], dtype=torch.float)
    
    return TrainingItem(x=tokens, y=grade)
    

train_data = HumicroeditTask1Dataset(split="train", transform=transform)
train_data = DataLoader(
    train_data, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    pin_memory=True, 
    drop_last=True)

dev_data = HumicroeditTask1Dataset(split="dev", transform=transform)
dev_data = DataLoader(
    dev_data, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    pin_memory=True, 
    drop_last=False)

bert_model = bert_model.to(device)
model = model.to(device)


# TRAINING
for epoch_i in range(EPOCHS):
    for t, (x, y) in enumerate(train_data):
        y = y.to(device)
        x = x.to(device)

        with torch.no_grad():
            bert_out = bert_model(x).pooler_output

        y_pred = model(bert_out) * 3.0

        # Compute and print loss.
        loss = loss_fn(y_pred, y)
        if t % 50 == 49:
            print(f"Training MSE of batch {t} is {loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    dev_losses = torch.zeros(len(dev_data), dtype=torch.float, device=device)
    for t, (x, y) in enumerate(dev_data):
        y = y.to(device)
        x = x.to(device)

        with torch.no_grad():
            bert_out = bert_model(x).pooler_output
            y_pred = model(bert_out) * 3.0

            loss = loss_fn(y_pred, y)
            dev_losses[t] = loss
            if t % 50 == 49:
                print(f"Dev MSE at batch {t} is {loss.item()}")
                
    print(f"Average dev RMSE is {torch.sqrt(dev_losses.mean())}")


test_data = HumicroeditTask1Dataset(split="test", transform=transform)
test_data = DataLoader(
    test_data, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    pin_memory=True, 
    drop_last=True)

test_losses = torch.zeros(len(test_data), dtype=torch.float, device=device)
for t, (x, y) in enumerate(test_data):
    y = y.to(device)
    x = x.to(device)

    with torch.no_grad():
        bert_out = bert_model(x).pooler_output
        y_pred = model(bert_out) * 3.0

        loss = loss_fn(y_pred, y)
        test_losses[t] = loss
        if t % 50 == 49:
            print(f"Test MSE at batch {t} is {loss.item()}")
                
print(f"Test RMSE is {torch.sqrt(test_losses.mean())}")