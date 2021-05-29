import torch
from torch import nn
import torch.nn.functional as F

from transformers import AutoModel

class RegressionModel(nn.Module):
    def __init__(self, transformer='bert-base-cased', word_emb_dim=0, amb_emb_dim=0):
        super(RegressionModel, self).__init__()

        self.sentence_embedder = AutoModel.from_pretrained(transformer).eval()
        num_features = self.sentence_embedder.pooler.dense.out_features + 2 * word_emb_dim + 4 * amb_emb_dim
        self.add_word_embs = bool(word_emb_dim)
        self.add_amb_embs = bool(amb_emb_dim)

        name_parts = [transformer]
        if word_emb_dim:
            name_parts += ['word-emb']
        if amb_emb_dim:
            name_parts += ['amb-emb']
        self.name = '_'.join(name_parts)

        self.l1 = nn.Linear(num_features, 256)
        self.l2 = nn.Linear(256, 256)
        self.lout = nn.Linear(256, 1)

    def forward(self, input_ids, attention_mask, token_type_ids, grade, **kwargs):
        with torch.no_grad():
            sentence_emb = self.sentence_embedder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            ).pooler_output

        parts = [sentence_emb]
        if self.add_word_embs:
            parts.extend(kwargs['word_ini_emb'], kwargs['word_fin_emb'])
        if self.add_amb_embs:
            parts.extend([kwargs['amb_emb_ini'],
                          kwargs['amb_mask_ini'],
                          kwargs['amb_emb_fin'],
                          kwargs['amb_mask_fin'],])
        features = torch.cat(parts, 1)
        # print(features.shape)

        x = F.relu(self.l1(features))
        x = F.relu(self.l2(x))
        y_pred = self.lout(x).squeeze(-1)
        loss = torch.sqrt(F.mse_loss(y_pred, grade))
        return loss, y_pred
