import re
import numpy as np
import math

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

def get_preprocess_ds(glove=None, synset_sizes=None, amb_feat=False, idx=''):
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
            f'hl_ini': hl_ini,
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
        if amb_feat:
            assert synset_sizes is not None
            def get_amb_feat(hl):
                tokens = hl.split()
                sizes = [synset_sizes[w] for w in tokens]
                size_prod = np.prod(sizes) if np.prod(sizes) != 0 else 0.0001
                f = math.log(size_prod)
                return [f]
            for pref in ['ini', 'fin']:
                feat = get_amb_feat(d[f'hl_{pref}{idx}'])
                d.update({
                    f'amb_feat_{pref}{idx}': feat
                })
        elif synset_sizes is not None:
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
