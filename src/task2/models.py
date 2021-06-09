import torch
import torch.nn as nn
from task1.models import RegressionModel

class GradeComparisonModel(nn.Module):
    def __init__(self, checkpoint_path, **regression_model_params):
        super(GradeComparisonModel, self).__init__()
        self.grader = RegressionModel(**regression_model_params)
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            # If the model is on the GPU, it still works!
            self.grader.load_state_dict(state_dict)

    def forward(self, labels, **regression_model_args):
        grades = {}
        for i in range(2):
            _, grades[i+1] = self.grader(**{k[:-1]: v for k, v in regression_model_args
                                            if k.endswith(str(i+1))})
        preds = grades[1] < grades[2]
        preds = torch.stack([~preds, preds], dim=1).float()
        return torch.FloatTensor([0]), preds
