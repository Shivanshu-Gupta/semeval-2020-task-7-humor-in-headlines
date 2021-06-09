import os
import re
import json
import comet_ml
from comet_ml import ExistingExperiment, ConfusionMatrix
from argparse import ArgumentParser
from params import outputs_dir

def upload_confusion_matrices(task_id, search_name, search_key):
    search_dir = os.path.join(outputs_dir, f'ray_results/task{task_id}/{search_name}')
    for exp_dir in os.listdir(search_dir):
        if not exp_dir.startswith(search_key):
            continue
        exp_dir_path = os.path.join(search_dir, exp_dir)
        comet_dir = ''
        for f in os.listdir(exp_dir_path):
            if f.startswith('comet'):
                comet_dir = f
        if not comet_dir:
            continue
        comet_exp_key = comet_dir.split('-')[1]
        print(comet_exp_key)
        comet_exp = ExistingExperiment(previous_experiment=comet_exp_key)
        comet_dir_path = os.path.join(exp_dir_path, comet_dir)
        p = re.compile(r'confusion-matrix-epoch-(.*).json')
        for cm_file in os.listdir(comet_dir_path):
            epoch = int(p.findall(cm_file)[0])
            cm_json = json.load(open(os.path.join(comet_dir_path, cm_file)))
            comet_exp._log_asset_data(data=cm_json, file_name=cm_file,
                                      overwrite=True, epoch=epoch,
                                      asset_type='confusion-matrix')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--task', type=int, default=1)
    parser.add_argument('--search_name', type=str, default='model1_base')
    parser.add_argument('--search_key', type=str, required=True)
    args = parser.parse_args()
    upload_confusion_matrices(args.task, args.search_name, args.search_key)
