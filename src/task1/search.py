import os
import sys
import comet_ml

# Better to set CUDA_VISIBLEDEVICES while running the script as: `CUDA_VISIBLE_DEVICES=0 python task1-driver.py <options>``
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['COMET_PROJECT_NAME'] = 'humor-1'
os.environ['COMET_MODE'] = 'ONLINE'

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_folder)
print(sys.path)

from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

from task1.util import setup
from task1.params import Task1Arguments, get_args, get_hp_space
from util import get_common_argparser, print_ds_stats, output as print
from params import outputs_dir

def main(cmd_args):
    args: Task1Arguments = get_args(cmd_args=cmd_args, search=True,
                                    skip_memory_metrics=True,
                                    disable_tqdm=True)
    tokenizer, ds, model_init, trainer = setup(args, search=True)
    print_ds_stats(ds, silent=cmd_args.silent)

    hp_space, is_grid = get_hp_space(cmd_args)
    print(hp_space(None))

    scheduler = ASHAScheduler(
        max_t=args.num_train_epochs,
        grace_period=5,
        reduction_factor=2)

    search_alg = None
    if cmd_args.usehyperopt:
        # current_best_params = [get_best_config(args.dataset, args.model, args.emb, args.enc)]
        current_best_params = []
        print(f'Using hyperopt with best guesses: {current_best_params}')
        search_alg = HyperOptSearch(metric="eval_rmse", mode="min",
                                    random_state_seed=args.seed,
                                    points_to_evaluate=current_best_params)
    if is_grid:
        cmd_args.num_samples = 1

    tune_args = dict(
        name=f'task1/{cmd_args.hpspace}',
        local_dir=os.path.join(outputs_dir, 'ray_results'),
        resources_per_trial={"cpu": 2, 'gpu': cmd_args.gpus_per_trial},
        scheduler=scheduler,
        search_alg=search_alg,
        log_to_file=True,
        metric='eval_rmse',
        mode='min'
    )
    best_run = trainer.hyperparameter_search(direction="minimize",
                                             hp_space=hp_space,
                                             compute_objective=lambda metrics: metrics['eval_rmse'],
                                             n_trials=cmd_args.num_samples,
                                             backend="ray",
                                             **tune_args)
    print(best_run)
    if cmd_args.evaluate_best:
        trainer.args.disable_tqdm = False
        for n, v in best_run.hyperparameters.items():
            setattr(trainer.args, n, v)

        trainer.train()
        eval_metrics = trainer.evaluate()
        print(eval_metrics)
        test_metrics = trainer.evaluate(ds['test'], metric_key_prefix='test')
        print(test_metrics)

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
