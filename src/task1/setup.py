import comet_ml
from dataclasses import asdict
from datasets.dataset_dict import DatasetDict
from transformers import AutoTokenizer, Trainer, EvalPrediction
from sklearn.metrics import mean_squared_error

from task1.data import get_dataset
from task1.models import RegressionModel
from task1.params import Task1Arguments
from params import MetricParams
from util import output as print

def compute_metrics(pred: EvalPrediction):
    grades = pred.label_ids
    preds = pred.predictions
    rmse = mean_squared_error(y_true=grades, y_pred=preds, squared=False)
    return dict(rmse=rmse)

def get_model_init(ds: DatasetDict):
    #_word_emb_dim = ds['train'][0]['word_ini_emb'].shape[0]
    #_amb_emb_dim = ds['train'][0]['amb_emb_ini'].shape[0]
    def model_init(args):
        if args is None: # required because Trainer seems to call model_init() in its constructor without arguments
            args = asdict(Task1Arguments(output_dir=''))
        word_emb_dim = 0
        if args['add_word_embs']:
            word_emb_dim = ds['train'][0]['word_ini_emb'].shape[0]
        amb_emb_dim = 0
        if args['add_amb_embs']:
            amb_emb_dim = ds['train'][0]['amb_emb_ini'].shape[0]
        amb_feat_dim = 2 if args['add_amb_feat'] else 0
        return RegressionModel(transformer=args['transformer'],
                                freeze_transformer=args['freeze_transformer'],
                                word_emb_dim=word_emb_dim,
                                amb_emb_dim=amb_emb_dim,
                                amb_feat_dim=amb_feat_dim)
    return model_init

def setup(args: Task1Arguments, search=False):
    tokenizer = AutoTokenizer.from_pretrained(args.transformer)
    ds = get_dataset(tokenizer=tokenizer,
                    add_word_embs=args.add_word_embs if not search else True,
                    add_amb_embs=args.add_amb_embs if not search else True,
                    add_amb_feat=args.add_amb_feat if not search else True,
                    hl_w_mod=False)
    model_init = get_model_init(ds)
    trainer_args = dict(
        args=args,
        train_dataset=ds['train'],
        eval_dataset=ds['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    metric = MetricParams(name='rmse', direction='minimize')
    if not search:
        model = model_init(asdict(args))
        trainer: Trainer = Trainer(model=model, **trainer_args)
        return tokenizer, ds, model, trainer, metric
    else:
        trainer: Trainer = Trainer(model_init=model_init, **trainer_args)
        return tokenizer, ds, model_init, trainer, metric
