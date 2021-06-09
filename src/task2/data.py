import os
import torchtext

import datasets
from datasets import load_dataset, ClassLabel
from datasets.dataset_dict import DatasetDict

from data import get_synsets_sizes, get_preprocess_ds

def add_T5_input(example):
    hl_ini = example['hl_ini']
    hl_fin1 = example['hl_fin1']
    hl_fin2 = example['hl_fin2']
    t5_input = f'headline humor: original: {hl_ini} edit1: {hl_fin1} edit2: {hl_fin2}'
    return {'t5_input': t5_input}

def get_encode(tokenizer, model_id=0):
    if model_id == 0:
        def encode(examples):
            d = {}
            for idx in range(2):
                _d = tokenizer(examples[f'hl_ini'], examples[f'hl_fin{idx+1}'],
                                truncation='longest_first', return_token_type_ids=True)
                d.update({f'{k}{idx+1}': v for k, v in _d.items()})
            return d
    elif model_id == 1:
        def encode(examples):
            return tokenizer(examples['hl_w_mod1'], examples['hl_w_mod2'],
                            truncation='longest_first', return_token_type_ids=True)
    elif model_id == 2:
        def encode(examples):
            return tokenizer(examples['t5_input'], truncation=True)
    return encode

# original1, edit1, original2, edit2 -> combined, hl_old1, hl_new1, hl_w_mod1, word_old1, word_new1, hl_old2, hl_new2, hl_w_mod2, word_old2, word_new2
def get_dataset(tokenizer, model_id=0, args=None, output_all_cols=False, data_dir=''):
    ds_path = os.path.join(data_dir, f'task2/preprocessed_data/model{model_id}', args.transformer)
    print(f'Dataset path: {ds_path}')
    try:
        encoded_ds = DatasetDict.load_from_disk(ds_path)
        print('Reloaded persisted dataset.')
    except:
        ds: DatasetDict = load_dataset("humicroedit", "subtask-2")
        glove, synset_sizes = None, None
        if model_id == 0:
            glove = torchtext.vocab.GloVe(name='840B', dim=300,
                                          cache=os.path.join(os.environ['HOME'], '.vector_cache'))
            synset_sizes = get_synsets_sizes(ds, task=2)

        for i in range(2):
            ds = ds.rename_column(f'edit{i+1}', f'word_fin{i+1}')
            ds = ds.map(get_preprocess_ds(glove=glove, synset_sizes=synset_sizes, idx=i+1))
            ds = ds.remove_columns([f'original{i+1}'])
            ds = ds.rename_column(f'meanGrade{i+1}', f'grade{i+1}')

        if model_id == 2:
            ds = ds.map(add_T5_input)

        ds = ds.rename_column('label', 'labels')
        binary_ds = ds.filter(lambda ex: ex['labels'] != 0).\
            map(lambda ex: {'labels': ex['labels'] - 1})
        binary_ds_features = ds['train'].features.copy()
        binary_ds_features['labels'] = ClassLabel(names=ds['train'].features['labels'].names[1:])
        binary_ds = binary_ds.cast(binary_ds_features)

        encode_fn = get_encode(tokenizer, model_id=model_id)
        encoded_ds = binary_ds.map(encode_fn, batched=True, batch_size=100)

        print('Saving preprocessed dataset.')
        os.makedirs(ds_path)
        encoded_ds.save_to_disk(ds_path)

    if model_id == 0:
        from task1.data import get_encoded_ds_cols
        encoded_ds_cols = get_encoded_ds_cols(args)
        encoded_ds_cols = [f'{col}{i+1}' for i in range(2) for col in encoded_ds_cols]
        encoded_ds_cols += ['grade1', 'grade2']
    elif model_id == 1 and args.transformer != 'distilbert-base-cased':
        encoded_ds_cols = ['input_ids', 'token_type_ids', 'attention_mask']
    else:
        encoded_ds_cols = ['input_ids', 'attention_mask']

    for _ds in encoded_ds.values():
        _ds.set_format(type='torch', columns=encoded_ds_cols + ['labels'],
                    output_all_columns=output_all_cols)

    return encoded_ds
