import torch
import torch.nn as nn
from task1.models import RegressionModel
from transformers.data.data_collator import DataCollatorWithPadding

class GradeComparisonDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        # import pdb; pdb.set_trace()
        batch = {}
        for i in range(2):
            _features = [{k[:-1]: v for k, v in feats.items()
                          if k.endswith(f'{i+1}')}
                         for feats in features]
            _batch = self.tokenizer.pad(
                _features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            batch.update({f'{k}{i+1}': v for k, v in _batch.items()})
        batch['labels'] = torch.Tensor([feats['labels'] for feats in features]).int()
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch

class GradeComparisonModel(nn.Module):
    def __init__(self, checkpoint_path, **regression_model_params):
        super(GradeComparisonModel, self).__init__()
        self.grader = RegressionModel(**regression_model_params)
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            # If the model is on the GPU, it still works!
            load_result = self.grader.load_state_dict(state_dict)
            print(load_result)
            if len(load_result.missing_keys) != 0:
                if load_result.missing_keys == self.grader._keys_to_ignore_on_save:
                    self.grader.tie_weights()
                else:
                    print(f"There were missing keys in the checkpoint model loaded: {load_result.missing_keys}.")
            if len(load_result.unexpected_keys) != 0:
                print(f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}.")

    def forward(self, **regression_model_args):
        grades = {}
        for i in range(2):
            _, grades[i+1] = self.grader(**{k[:-1]: v for k, v in regression_model_args.items()
                                            if k.endswith(f'{i+1}')})
        logits = grades[1] < grades[2]
        logits = torch.stack([~logits, logits], dim=1).float()
        return torch.FloatTensor([0]), logits
