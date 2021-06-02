import comet_ml
from dataclasses import asdict
from datasets.dataset_dict import DatasetDict
from transformers import AutoTokenizer, Trainer

from task1.data import get_dataset
from task1.models import RegressionModel
from task1.metrics import get_compute_metrics
from task1.params import Task1Arguments
from util import output as print

def get_model_init(ds: DatasetDict):
    _word_emb_dim = ds['train'][0]['word_ini_emb'].shape[0]
    _amb_emb_dim = ds['train'][0]['amb_emb_ini'].shape[0]
    def model_init(args):
        if args:
            # args = Task1Arguments(**args)
            word_emb_dim = _word_emb_dim if args['add_word_embs'] else 0
            amb_emb_dim = _amb_emb_dim if args['add_amb_embs'] else 0
            return RegressionModel(transformer=args['transformer'],
                                   freeze_transformer=args['freeze_transformer'],
                                   word_emb_dim=word_emb_dim,
                                   amb_emb_dim=amb_emb_dim)
        else: # required because Trainer seems to call model_init() in its constructor without arguments
            return RegressionModel()
    return model_init

def setup(args: Task1Arguments, search=False):
    tokenizer = AutoTokenizer.from_pretrained(args.transformer)
    ds = get_dataset(tokenizer=tokenizer,
                    add_word_embs=args.add_word_embs if not search else True,
                    add_amb_embs=args.add_amb_embs if not search else True,
                    hl_w_mod=False)
    model_init = get_model_init(ds)
    if not search:
        model = model_init(asdict(args))
        trainer: Trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds['train'],
            eval_dataset=ds['validation'],
            tokenizer=tokenizer,
            compute_metrics=get_compute_metrics()
        )
        return tokenizer, ds, model, trainer
    else:
        trainer: Trainer = Trainer(
            model_init=model_init,
            args=args,
            train_dataset=ds['train'],
            eval_dataset=ds['validation'],
            tokenizer=tokenizer,
            compute_metrics=get_compute_metrics()
        )
        return tokenizer, ds, model_init, trainer
