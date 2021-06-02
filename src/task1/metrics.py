from sklearn.metrics import mean_squared_error
from transformers import EvalPrediction

def get_compute_metrics():
    def compute_metrics(pred: EvalPrediction):
        grades = pred.label_ids
        preds = pred.predictions
        mse = mean_squared_error(y_true=grades, y_pred=preds, squared=False)
        return {
            'rmse': mse
        }
    return compute_metrics