import torch
from torch import nn
import torch.nn.functional as F

from transformers import AutoModel

class RegressionModelv1(nn.Module):
    def __init__(self, transformer='bert-base-cased'):
        super(RegressionModelv1, self).__init__()

        self.sentence_embedder = AutoModel.from_pretrained(transformer).eval()

        self.l1 = nn.Linear(768, 256)
        self.l2 = nn.Linear(256, 256)
        self.lout = nn.Linear(256, 1)

    def forward(self, input_ids, attention_mask, token_type_ids, grade):
        with torch.no_grad():
            sentence_emb = self.sentence_embedder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            ).pooler_output

        x = F.relu(self.l1(sentence_emb))
        x = F.relu(self.l2(x))
        y_pred = self.lout(x).squeeze(-1)
        loss = F.mse_loss(y_pred, grade)
        return loss, y_pred

class RegressionModelv2(nn.Module):
    def __init__(self, transformer='bert-base-cased', word_emb_dim=300):
        super(RegressionModelv2, self).__init__()

        self.sentence_embedder = AutoModel.from_pretrained(transformer).eval()

        num_features = self.sentence_embedder.pooler.dense.out_features + 2 * word_emb_dim

        self.l1 = nn.Linear(num_features, 256)
        self.l2 = nn.Linear(256, 256)
        self.lout = nn.Linear(256, 1)

    def forward(self, input_ids, attention_mask, token_type_ids, word_ini_emb, word_fin_emb, grade):
        with torch.no_grad():
            sentence_emb = self.sentence_embedder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            ).pooler_output

        features = torch.cat((word_ini_emb, sentence_emb, word_fin_emb), 1)

        x = F.relu(self.l1(features))
        x = F.relu(self.l2(x))
        y_pred = self.lout(x).squeeze(-1)
        loss = F.mse_loss(y_pred, grade)
        return loss, y_pred
