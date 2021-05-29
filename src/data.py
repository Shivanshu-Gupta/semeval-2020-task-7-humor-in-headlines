import re
import numpy as np
import torchtext

import datasets
from datasets import load_dataset, ClassLabel

# def _make_headline(s, e):
#     p = re.compile(r'<(.*)/>')
#     return p.sub(f'[\\1|{e}]', s)

def get_synsets_sizes(ds, task=1):
    from nltk.corpus import wordnet
    synset_sizes = {}
    p = re.compile(r'<(.*)/>')
    for split in ds:
        for sample in ds[split]:
            if task == 1: idxs = ['']
            else: idxs = ['1', '2']
            for idx in idxs:
                original = p.sub(f'\\1', sample[f'original{idx}'])
                words = list(original.split()) + [sample[f'edit{idx}']]
                for w in words:
                    if w not in synset_sizes:
                        synset_sizes[w] = len(wordnet.synsets(w))
    return synset_sizes

def get_preprocess_ds(glove=None, synset_sizes=None, idx=''):
    if synset_sizes is not None:
        max_len = max(synset_sizes.values())
    def preprocess_ds(example):
        hl = example[f'original{idx}']
        word_fin = example[f'word_fin{idx}']

        opening = hl.index("<")
        closing = hl.index("/>", opening)

        prefix = hl[:opening]
        suffix = hl[closing + 2:]
        word_ini = hl[opening+1:closing]

        hl_ini = prefix + word_ini + suffix
        hl_fin = prefix + word_fin + suffix
        hl_w_mod = prefix + f'[{word_ini} | {word_fin}]' + suffix
        d = {
            f'hl_ini{idx}': hl_ini,
            f'hl_fin{idx}': hl_fin,
            f'word_ini{idx}': word_ini,
            f'hl_w_mod{idx}': hl_w_mod,
        }
        if glove is not None:
            words = [word_ini, word_fin]
            word_embs = glove.get_vecs_by_tokens(words, lower_case_backup=True)
            d.update({
                f'word_ini{idx}_emb': word_embs[0].numpy(),
                f'word_fin{idx}_emb': word_embs[1].numpy()
            })
        if synset_sizes is not None:
            def get_amb_emb(hl):
                tokens = hl.split()
                amb_emb = np.zeros([max_len])
                amb_emb[:len(tokens)] = [synset_sizes[w] for w in tokens]
                amb_mask = np.zeros([max_len])
                amb_mask[:len(tokens)] = 1
                return amb_emb, amb_mask
            for pref in ['ini', 'fin']:
                amb_emb, amb_mask = get_amb_emb(d[f'hl_{pref}{idx}'])
                d.update({
                    f'amb_emb_{pref}{idx}': amb_emb,
                    f'amb_mask_{pref}{idx}': amb_mask,
                })
        return d

    return preprocess_ds

def get_encode1(tokenizer, hl_w_mod=False):
    if hl_w_mod:
        def encode(examples):
            return tokenizer(examples['hl_w_mod'], truncation=True)
    else:
        def encode(examples):
            return tokenizer(examples['hl_ini'], examples['hl_fin'], truncation='longest_first')
    return encode

def get_encode2(tokenizer, combined=True, hl_w_mod=False):
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


# original, edit -> hl_old, hl_new, hl_mod, word_old, word_new
def get_task1_dataset(tokenizer, add_word_embs=False, add_amb_embs=False,
                      hl_w_mod=False, output_all_cols=False):
    ds: datasets.DatasetDict = load_dataset("humicroedit", "subtask-1")

    glove = torchtext.vocab.GloVe(name='840B', dim=300) if add_word_embs else None
    synset_sizes = get_synsets_sizes(ds) if add_amb_embs else None

    ds = ds.rename_column('edit', 'word_fin')
    ds = ds.map(get_preprocess_ds(glove=glove, synset_sizes=synset_sizes))
    ds = ds.remove_columns(['original'])

    ds = ds.rename_column('meanGrade', 'grade')
    encode_fn = get_encode1(tokenizer, hl_w_mod=hl_w_mod)
    encoded_ds = ds.map(encode_fn, batched=True, batch_size=100)
    encoded_ds_cols = ['input_ids', 'token_type_ids', 'attention_mask']
    if add_word_embs:
        encoded_ds_cols.extend(['word_ini_emb', 'word_fin_emb'])
    if add_amb_embs:
        encoded_ds_cols.extend(['amb_emb_ini', 'amb_mask_ini',
                                'amb_emb_fin', 'amb_mask_fin'])
    for _ds in encoded_ds.values():
        _ds.set_format(type='torch', columns=encoded_ds_cols + ['grade'],
                       output_all_columns=output_all_cols)
    return encoded_ds

# original1, edit1, original2, edit2 -> combined, hl_old1, hl_new1, hl_w_mod1, word_old1, word_new1, hl_old2, hl_new2, hl_w_mod2, word_old2, word_new2
def get_task2_dataset(tokenizer, add_word_embs=False, add_amb_embs=False,
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

    encode_fn = get_encode2(tokenizer, hl_w_mod=hl_w_mod, combined=combined)
    encoded_ds = binary_ds.map(encode_fn, batched=True, batch_size=100)

    encoded_ds_cols = ['input_ids', 'token_type_ids', 'attention_mask']
    if add_word_embs:
        encoded_ds_cols.extend(['word_ini_emb', 'word_fin_emb'])
    if add_amb_embs:
        encoded_ds_cols.extend(['amb_emb_ini', 'amb_mask_ini',
                                'amb_emb_fin', 'amb_mask_fin'])
    encoded_ds_cols = [f'{col}{i+1}' for i in range(2) for col in encoded_ds_cols]

    for _ds in encoded_ds.values():
        _ds.set_format(type='torch', columns=encoded_ds_cols + ['labels'],
                    output_all_columns=output_all_cols)

    return encoded_ds
