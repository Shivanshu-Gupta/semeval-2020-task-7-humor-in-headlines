import os
import sys
import argparse
import comet_ml
import importlib

# Better to set CUDA_VISIBLEDEVICES while running the script as: `CUDA_VISIBLE_DEVICES=0 python task1-driver.py <options>``
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['COMET_PROJECT_NAME'] = 'humor-1'
os.environ['COMET_MODE'] = 'ONLINE'

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_folder)
print(sys.path)

from ray.tune import ASHAScheduler, HyperOptSearch

from task1 import params
from task1.util import setup
from task1.params import Task1Arguments, get_args
from util import get_common_argparser, print_ds_stats, output as print
from params import ray_dir

def main(cmd_args):
    args: Task1Arguments = get_args(cmd_args=cmd_args, search=True)
    tokenizer, ds, model_init, trainer = setup(args, search=True)
    print_ds_stats(ds, silent=cmd_args.silent)

    def objective_fn(metrics):
        return metrics['rmse']

    hp_space, is_grid = getattr(params, f'hp_space_{cmd_args.hpspace}')(hyperopt=cmd_args.usehyperopt)

    scheduler = ASHAScheduler(
        max_t=args.num_train_epochs,
        grace_period=5,
        reduction_factor=2)

    # Specify the search space and maximize score
    search_alg = None
    if cmd_args.usehyperopt:
        # current_best_params = [get_best_config(args.dataset, args.model, args.emb, args.enc)]
        # print(f'Using hyperopt with best guesses: {current_best_params}')
        search_alg = HyperOptSearch(metric="val_accuracy", mode="max",
                                    random_state_seed=args.seed,
                                    # points_to_evaluate=current_best_params
                                    )
    elif is_grid:
        cmd_args.num_samples = 1
    tune_args = dict(
        name=f'task1/{cmd_args.hpspace}',
        local_dir=ray_dir,
        resources_per_trial={"cpu": 2, 'gpu': cmd_args.gpus_per_trial},
        scheduler=scheduler,
        search_alg=search_alg,
        log_to_file=True,
        mode='min'
    )
    best_run = trainer.hyperparameter_search(direction="minimize",
                                             hp_space=hp_space,
                                             compute_objective=objective_fn,
                                             n_trials=cmd_args.num_samples,
                                             **tune_args)
    print(best_run)
    for n, v in best_run.hyperparameters.items():
        setattr(trainer.args, n, v)

    trainer.train()
    trainer.evaluate()

if __name__ == '__main__':
    parser = get_common_argparser()
    parser.add_argument('--hpspace', type=str, default='1')
    parser.add_argument('--usehyperopt', action='store_true')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--gpus_per_trial', type=int, default=1)
    parser.add_argument('--evaluate_best', action='store_true')
    cmd_args = parser.parse_args()

    print(cmd_args, silent=cmd_args.silent)
    main(cmd_args)
