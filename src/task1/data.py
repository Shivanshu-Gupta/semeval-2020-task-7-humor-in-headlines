import os
import torchtext

import datasets
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict

from data import get_synsets_sizes, get_preprocess_ds

def get_encode(tokenizer):
    def encode(examples):
        return tokenizer(examples['hl_ini'], examples['hl_fin'], truncation='longest_first',
                         return_token_type_ids=True)
    return encode

def get_encoded_ds_cols(args):
    if args.transformer != 'distilbert-base-cased':
        encoded_ds_cols = ['input_ids', 'token_type_ids', 'attention_mask']
    else:
        encoded_ds_cols = ['input_ids', 'attention_mask']
    if args.add_word_embs:
        encoded_ds_cols.extend(['word_ini_emb', 'word_fin_emb'])
    if args.add_amb_embs:
        encoded_ds_cols.extend(['amb_emb_ini', 'amb_mask_ini',
                                'amb_emb_fin', 'amb_mask_fin'])
    if args.add_amb_feat:
        encoded_ds_cols.extend(['amb_feat_ini','amb_feat_fin'])

    return encoded_ds_cols

# original, edit -> hl_ini, hl_fin, hl_mod, word_ini, word_fin
def get_dataset(tokenizer, args, output_all_cols=False, data_dir=''):
    ds_path = os.path.join(data_dir, f'task1/preprocessed_data', args.transformer)
    print(f'Dataset path: {ds_path}')
    try:
        encoded_ds = DatasetDict.load_from_disk(ds_path)
        print('Reloaded persisted dataset.')
    except:
        ds: DatasetDict = load_dataset("humicroedit", "subtask-1")
        glove = torchtext.vocab.GloVe(name='840B', dim=300,
                                      cache=os.path.join(os.environ['HOME'], '.vector_cache'))
        synset_sizes = get_synsets_sizes(ds)

        ds = ds.rename_column('edit', 'word_fin')
        ds = ds.map(get_preprocess_ds(glove=glove, synset_sizes=synset_sizes, add_amb_feat=True))
        ds = ds.remove_columns(['original'])

        ds = ds.rename_column('meanGrade', 'grade')
        encode_fn = get_encode(tokenizer)
        encoded_ds = ds.map(encode_fn, batched=True, batch_size=100)

        print('Saving preprocessed dataset.')
        os.makedirs(ds_path)
        encoded_ds.save_to_disk(ds_path)

    encoded_ds_cols = get_encoded_ds_cols(args)
    for _ds in encoded_ds.values():
        _ds.set_format(type='torch', columns=encoded_ds_cols + ['grade'],
                       output_all_columns=output_all_cols)
    return encoded_ds
