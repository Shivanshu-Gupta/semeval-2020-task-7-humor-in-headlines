# run: `python search.py --task <task_id> <options>``
# python search.py --task 1 -o --comet --num_epochs 3 --gpus_per_trial 1 --evaluate_best
import os
import comet_ml
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

from util import get_common_argparser, print_ds_stats, output as print
from params import outputs_dir

os.environ['COMET_DISABLE_AUTO_LOGGING']='1'    # temporarily to prevent: ImportError: You must import Comet before these modules: torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['COMET_MODE'] = 'ONLINE'

def main(cmd_args):
    task_id = cmd_args.task
    os.environ['COMET_PROJECT_NAME'] = f'humor-{task_id}'
    if task_id == 1:
        from task1.params import Task1Arguments as TaskArguments, get_args, get_hp_space
        from task1.setup import setup
    else:
        from task2.params import Task2Arguments as TaskArguments, get_args, get_hp_space
        from task2.setup import setup

    args: TaskArguments = get_args(cmd_args=cmd_args, search=True,
                                skip_memory_metrics=True,   # without this trainer.hyperparameter_search() + raytune bugs out
                                disable_tqdm=True)
    tokenizer, ds, model_init, trainer, metric = setup(args, search=True)
    print_ds_stats(ds, silent=silent)

    hp_space, is_grid = get_hp_space(cmd_args)
    print(hp_space(None), silent=silent)

    scheduler = ASHAScheduler(
        max_t=args.num_train_epochs,
        grace_period=min(5, args.num_train_epochs),
        reduction_factor=2)

    search_alg = None
    if cmd_args.hyperopt:
        current_best_params = []
        print(f'Using hyperopt with best guesses: {current_best_params}', silent=silent)
        search_alg = HyperOptSearch(metric=f"eval_{metric.name}", mode="min",
                                    random_state_seed=args.seed,
                                    points_to_evaluate=current_best_params)
    if is_grid:
        cmd_args.num_samples = 1

    tune_args = dict(
        name=f'task{task_id}/{cmd_args.hpspace}',
        local_dir=os.path.join(outputs_dir, 'ray_results'),
        resources_per_trial={"cpu": 2, 'gpu': cmd_args.gpus_per_trial},
        scheduler=scheduler,
        search_alg=search_alg,
        log_to_file=True,
        metric=f'eval_{metric.name}',
        mode=metric.mode
    )
    best_run = trainer.hyperparameter_search(direction=metric.direction,
                                            hp_space=hp_space,
                                            compute_objective=lambda metrics: metrics['eval_rmse'],
                                            n_trials=cmd_args.num_samples,
                                            backend="ray",
                                            **tune_args)
    print(best_run, silent=silent)
    if cmd_args.evaluate_best:
        trainer.args.disable_tqdm = False
        for n, v in best_run.hyperparameters.items():
            setattr(trainer.args, n, v)

        trainer.train()
        eval_metrics = trainer.evaluate()
        print(eval_metrics, silent=silent)
        if task_id == 2:    # confusion matrix in task 2's compute_metrics needs index_to_example_function
            from task2.setup import get_compute_metrics
            label_names = ds['train'].features['labels'].names
            trainer.compute_metrics = get_compute_metrics(tokenizer=tokenizer, ds=ds,
                                                        label_names=label_names, split='test')
        test_metrics = trainer.evaluate(ds['test'], metric_key_prefix='test')
        print(test_metrics, silent=silent)

if __name__ == '__main__':
    parser = get_common_argparser()
    parser.add_argument('--task', type=int, choices=[1, 2], default=1)
    parser.add_argument('--hpspace', type=str, default='1')
    parser.add_argument('--hyperopt', action='store_true')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--gpus_per_trial', type=int, default=0)
    parser.add_argument('--evaluate_best', action='store_true')
    cmd_args = parser.parse_args()
    silent = cmd_args.silent
    print(cmd_args, silent=silent)

    main(cmd_args)
