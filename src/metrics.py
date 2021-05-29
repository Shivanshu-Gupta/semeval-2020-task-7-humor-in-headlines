import comet_ml
import numpy as np
from pdb import set_trace

from transformers import EvalPrediction
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error

def compute_metrics_task1(pred: EvalPrediction):
    grades = pred.label_ids
    preds = pred.predictions
    # set_trace()
    # rmse = mean_squared_error(y_true=grades, y_pred=preds, squared=False)
    rmse = np.sqrt(np.mean((grades - preds)**2))            # taken from code/score_task1.py
    return {'rmse': rmse}

def get_compute_metrics_task2(tokenizer, ds, label_names):
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
                                        index_to_example_function=lambda idx: tokenizer.decode(ds['validation'][idx]['input_ids']))
            cm.display()
            # experiment.log_confusion_matrix(matrix=cm, file_name=f"confusion-matrix-epoch-{epoch}.json")

        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    return compute_metrics