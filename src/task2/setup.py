import comet_ml
from dataclasses import asdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets.dataset_dict import DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, EvalPrediction

from task2.data import get_dataset
from task2.params import Task2Arguments
from params import MetricParams
from util import output as print

def get_compute_metrics(tokenizer, ds, label_names, split='validation'):
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
            cm = comet_ml.ConfusionMatrix(y_true=labels, y_predicted=preds, labels=label_names,
                                        title=f"Confusion Matrix, Epoch #{epoch}",
                                        index_to_example_function=lambda idx: tokenizer.decode(ds[split][idx]['input_ids']))
            print()
            cm.display()
            print()
            # experiment.log_confusion_matrix(matrix=cm, file_name=f"confusion-matrix-epoch-{epoch}.json")
        return dict(accuracy=acc, f1=f1, precision=precision, recall=recall)
    return compute_metrics

def model_init(args):
    if args is None:
        args = asdict(Task2Arguments())
    return AutoModelForSequenceClassification.from_pretrained(args['transformer'], num_labels=2)

def setup(args: Task2Arguments, search=False):
    tokenizer = AutoTokenizer.from_pretrained(args.transformer)
    ds: DatasetDict = get_dataset(tokenizer=tokenizer, combined=args.combined)
    label_names = ds['train'].features['labels'].names
    compute_metrics = get_compute_metrics(tokenizer=tokenizer, ds=ds, label_names=label_names)
    trainer_args = dict(
        args=args,
        train_dataset=ds['train'],
        eval_dataset=ds['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    metric = MetricParams(name='accuracy', direction='maximize')
    if not search:
        model = model_init(asdict(args))
        trainer: Trainer = Trainer(model=model, **trainer_args)
        return tokenizer, ds, model, trainer, metric
    else:
        trainer: Trainer = Trainer(model_init=model_init, **trainer_args)
        return tokenizer, ds, model_init, trainer, metric
