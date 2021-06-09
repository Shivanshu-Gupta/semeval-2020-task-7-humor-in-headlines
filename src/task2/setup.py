import os
import json
import comet_ml
from dataclasses import asdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets.dataset_dict import DatasetDict
from transformers import AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, EvalPrediction

from task2.data import get_dataset
from task2.models import GradeComparisonModel
from params import MetricParams
from util import output as print

def get_compute_metrics(tokenizer, ds, split='validation'):
    def compute_metrics(pred: EvalPrediction):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)

        experiment = comet_ml.get_global_experiment()
        if experiment:
            epoch = int(experiment.curr_epoch) if experiment.curr_epoch is not None else 0
            # experiment.set_epoch(epoch)
            # experiment.log_metrics(metrics)
            cm = comet_ml.ConfusionMatrix(y_true=labels, y_predicted=preds, labels=ds[split].features['labels'].names,
                                          title=f"Confusion Matrix, Epoch #{epoch}",
                                          index_to_example_function=lambda idx: tokenizer.decode(ds[split][idx]['input_ids']))
            comet_exp_key = experiment.get_key()
            json.dump(cm.to_json(), open(f'comet-{comet_exp_key}/confusion-matrix-epoch-{epoch}.json', 'w'))
            # experiment.log_confusion_matrix(matrix=cm, file_name=f"confusion-matrix-epoch-{epoch}.json")
        return dict(accuracy=acc, f1=f1, precision=precision, recall=recall)
    return compute_metrics

def get_model_init(args, ds: DatasetDict=None):
    model_id = args.model_id
    if model_id == 0:
        word_emb_dim = ds['train'][0]['word_ini_emb'].shape[0] if args.add_word_emb else 0
        amb_emb_dim = ds['train'][0]['amb_emb_ini'].shape[0] if args.add_word_emb else 0
        def model_init(trial):
            if trial is None: trial = asdict(args)  # required because Trainer seems to call model_init() in its constructor without arguments
            return GradeComparisonModel(checkpoint_path=None,
                                        transformer=trial['transformer'],
                                        freeze_transformer=trial['freeze_transformer'],
                                        word_emb_dim=word_emb_dim,
                                        amb_emb_dim=amb_emb_dim)
    elif model_id == 1:
        def model_init(trial):
            if trial is None: trial = asdict(args)
            return AutoModelForSequenceClassification.from_pretrained(trial['transformer'],
                                                                      num_labels=2)
            # return Model1(trial['transformer'])
    elif model_id == 2:
        def model_init(trial):
            if trial is None: trial = asdict(args)
            return AutoModelForSeq2SeqLM.from_pretrained(trial['transformer'])
    return model_init

def setup(args, data_dir=''):
    model_id = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(args.transformer)
    ds: DatasetDict = get_dataset(tokenizer=tokenizer, model_id=model_id,
                                  args=args, data_dir=data_dir)
    compute_metrics = get_compute_metrics(tokenizer=tokenizer, ds=ds)
    trainer_args = dict(
        args=args,
        train_dataset=ds['train'],
        eval_dataset=ds['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    metric = MetricParams(name='accuracy', direction='maximize')

    # use model init even if not doing hyperparameter_search for reproducibility
    model_init = get_model_init(args=args, ds=ds)
    trainer: Trainer = Trainer(model_init=model_init, **trainer_args)
    return tokenizer, ds, model_init, trainer, metric
