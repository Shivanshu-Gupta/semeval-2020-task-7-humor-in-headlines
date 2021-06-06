import torchtext

import datasets
from datasets import load_dataset

from data import get_synsets_sizes, get_preprocess_ds

def get_encode(tokenizer, hl_w_mod=False):
    if hl_w_mod:
        def encode(examples):
            return tokenizer(examples['hl_w_mod'], truncation=True)
    else:
        def encode(examples):
            return tokenizer(examples['hl_ini'], examples['hl_fin'], truncation='longest_first')
    return encode

# original, edit -> hl_old, hl_new, hl_mod, word_old, word_new
def get_dataset(tokenizer, add_word_embs=False, add_amb_embs=False, add_amb_feat=False,
                      hl_w_mod=False, output_all_cols=False):
    ds: datasets.DatasetDict = load_dataset("humicroedit", "subtask-1")

    glove = torchtext.vocab.GloVe(name='840B', dim=300) if add_word_embs else None
    synset_sizes = get_synsets_sizes(ds) if (add_amb_embs or add_amb_feat) else None

    ds = ds.rename_column('edit', 'word_fin')
    ds = ds.map(get_preprocess_ds(glove=glove, synset_sizes=synset_sizes, amb_feat=add_amb_feat))
    ds = ds.remove_columns(['original'])

    ds = ds.rename_column('meanGrade', 'grade')
    encode_fn = get_encode(tokenizer, hl_w_mod=hl_w_mod)
    encoded_ds = ds.map(encode_fn, batched=True, batch_size=100)
    encoded_ds_cols = ['input_ids', 'token_type_ids', 'attention_mask']
    if add_word_embs:
        encoded_ds_cols.extend(['word_ini_emb', 'word_fin_emb'])
    if add_amb_embs:
        encoded_ds_cols.extend(['amb_emb_ini', 'amb_mask_ini',
                                'amb_emb_fin', 'amb_mask_fin'])
    if add_amb_feat:
        encoded_ds_cols.extend(['amb_feat_ini','amb_feat_fin'])
    for _ds in encoded_ds.values():
        _ds.set_format(type='torch', columns=encoded_ds_cols + ['grade'],
                       output_all_columns=output_all_cols)
    return encoded_ds
