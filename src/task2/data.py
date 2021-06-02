import torchtext

import datasets
from datasets import load_dataset, ClassLabel

from data import get_synsets_sizes, get_preprocess_ds

def get_encode(tokenizer, combined=True, hl_w_mod=False):
    if combined:
        def encode(examples):
            return tokenizer(examples['hl_w_mod1'], examples['hl_w_mod2'],
                            truncation='longest_first')
    elif hl_w_mod:
        def encode(examples):
            d = {}
            for idx in range(2):
                _d = tokenizer(examples[f'hl_w_mod{idx+1}'],
                               truncation=True)
                d.update({f'{k}{idx+1}': v for k, v in _d.items()})
            return d
    else:
        def encode(examples):
            d = {}
            for idx in range(2):
                _d = tokenizer(examples[f'hl_ini{idx+1}'], examples[f'hl_fin{idx+1}'],
                               truncation='longest_first')
                d.update({f'{k}{idx+1}': v for k, v in _d.items()})
            return d
    return encode

# original1, edit1, original2, edit2 -> combined, hl_old1, hl_new1, hl_w_mod1, word_old1, word_new1, hl_old2, hl_new2, hl_w_mod2, word_old2, word_new2
def get_dataset(tokenizer, add_word_embs=False, add_amb_embs=False,
                      hl_w_mod=False, combined=True, output_all_cols=False):
    ds: datasets.DatasetDict = load_dataset("humicroedit", "subtask-2")
    glove = torchtext.vocab.GloVe(name='840B', dim=300) if add_word_embs else None
    synset_sizes = get_synsets_sizes(ds) if add_amb_embs else None

    for i in range(2):
        ds = ds.rename_column(f'edit{i+1}', f'word_fin{i+1}')
        ds = ds.map(get_preprocess_ds(glove=glove, synset_sizes=synset_sizes, idx=i+1))
        ds = ds.remove_columns([f'original{i+1}'])

    ds = ds.rename_column('label', 'labels')
    binary_ds = ds.filter(lambda ex: ex['labels'] != 0).\
        map(lambda ex: {'labels': ex['labels'] - 1})
    binary_ds_features = ds['train'].features.copy()
    binary_ds_features['labels'] = ClassLabel(names=ds['train'].features['labels'].names[1:])
    binary_ds = binary_ds.cast(binary_ds_features)

    encode_fn = get_encode(tokenizer, hl_w_mod=hl_w_mod, combined=combined)
    encoded_ds = binary_ds.map(encode_fn, batched=True, batch_size=100)

    tokenizer_cols = ['input_ids', 'token_type_ids', 'attention_mask']
    encoded_ds_cols = tokenizer_cols if not combined else []
    if add_word_embs:
        encoded_ds_cols.extend(['word_ini_emb', 'word_fin_emb'])
    if add_amb_embs:
        encoded_ds_cols.extend(['amb_emb_ini', 'amb_mask_ini',
                                'amb_emb_fin', 'amb_mask_fin'])
    encoded_ds_cols = [f'{col}{i+1}' for i in range(2) for col in encoded_ds_cols]
    if combined:
        encoded_ds_cols.extend(tokenizer_cols)

    for _ds in encoded_ds.values():
        _ds.set_format(type='torch', columns=encoded_ds_cols + ['labels'],
                    output_all_columns=output_all_cols)

    return encoded_ds
