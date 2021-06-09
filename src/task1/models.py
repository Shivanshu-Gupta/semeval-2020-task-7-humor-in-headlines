from contextlib import nullcontext

import torch
from torch import nn
import torch.nn.functional as F

from transformers import AutoModel

class RegressionModel(nn.Module):
    def __init__(self, transformer='bert-base-cased', freeze_transformer=True, word_emb_dim=0, amb_emb_dim=0, amb_feat_dim=0):
        super(RegressionModel, self).__init__()
        self.transformer = transformer
        self.freeze_transformer = freeze_transformer
        self.sentence_embedder = AutoModel.from_pretrained(transformer)
        if freeze_transformer:
            self.sentence_embedder = self.sentence_embedder.eval()
        transformer_output_size = 768   # true for {bert, roberta, distilbert, distilroberta}-base
        if transformer.endswith('large'): # required for {bert, roberta}-large
            transformer_output_size = self.sentence_embedder.pooler.dense.out_features
        num_features = transformer_output_size + 2 * word_emb_dim + 4 * amb_emb_dim + amb_feat_dim
        self.add_word_embs = bool(word_emb_dim)
        self.add_amb_embs = bool(amb_emb_dim)
        self.add_amb_feat = bool(amb_feat_dim)

        self.l1 = nn.Linear(num_features, 256)
        self.l2 = nn.Linear(256, 256)
        self.lout = nn.Linear(256, 1)

    def forward(self, input_ids, attention_mask, grade, **kwargs):
        with torch.no_grad() if self.freeze_transformer else nullcontext():
            if self.transformer != 'distilbert-base-cased':
                sentence_emb = self.sentence_embedder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=kwargs['token_type_ids']
                ).pooler_output
            else:
                sentence_emb = self.sentence_embedder(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).last_hidden_state[:, 0]
        parts = [sentence_emb]
        if self.add_word_embs:
            parts.extend([kwargs['word_ini_emb'], kwargs['word_fin_emb']])
        if self.add_amb_embs:
            parts.extend([kwargs['amb_emb_ini'],
                          kwargs['amb_mask_ini'],
                          kwargs['amb_emb_fin'],
                          kwargs['amb_mask_fin'],])
        if self.add_amb_feat:
            parts.extend([kwargs['amb_feat_ini'],
                          kwargs['amb_feat_fin']])
        features = torch.cat(parts, 1)
        # print(features.shape)

        x = F.relu(self.l1(features))
        x = F.relu(self.l2(x))
        y_pred = self.lout(x).squeeze(-1)
        loss = F.mse_loss(y_pred, grade)
        return loss, y_pred
