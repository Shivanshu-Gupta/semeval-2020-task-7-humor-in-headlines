# run: `python search.py --task <task_id> <options>``
# python search.py --task 1 --hp_space 1 -o --comet --num_epochs 3 --gpus_per_trial 1 --evaluate_best
# python search.py --task 2 --hp_space model1_0 -o --comet --num_epochs 3 --gpus_per_trial 1 --evaluate_best
import os
from dataclasses import asdict
import comet_ml

from transformers import set_seed
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import IntervalStrategy

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.analysis.experiment_analysis import ExperimentAnalysis

from util import get_common_argparser, print_ds_stats, output as print
from params import inputs_dir, outputs_dir, MetricParams
from comet_upload import upload_confusion_matrices

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['COMET_DISABLE_AUTO_LOGGING']='1'    # to prevent: ImportError: You must import Comet before these modules: torch
os.environ['TUNE_MAX_PENDING_TRIALS_PG'] = '16' # to prevent ray from filling up filling up disk space

class CometCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        experiment: comet_ml.Experiment = comet_ml.config.get_global_experiment()
        experiment.log_parameters(asdict(args))
        if args.search_name: experiment.add_tag(args.search_name)
        experiment.add_tag(args.transformer)
        comet_exp_key = experiment.get_key()
        os.makedirs(f'comet-{comet_exp_key}', exist_ok=True)

class ReportMetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        tune.report(iterations=int(state.epoch), **metrics)

def train(config, task_id, args, comet=False, checkpoint_dir=None):
    from datasets.utils.logging import set_verbosity_error
    set_verbosity_error()   # to not show ds map progress bars

    for n, v in config.items():
        setattr(args, n, v)
    print(args)

    if task_id == 1: from task1.setup import setup
    else: from task2.setup import setup
    set_seed(args.seed)
    tokenizer, ds, model_init, trainer, metric = setup(args, data_dir='../../..')

    trainer.args._n_gpu = 1
    trainer.add_callback(ReportMetricsCallback)
    if comet: trainer.add_callback(CometCallback)

    trainer.train()

def main(cmd_args):
    task_id = cmd_args.task
    if cmd_args.comet: setup_comet(task_id=task_id)
    ray.init(dashboard_host='0.0.0.0', _temp_dir='/srv/ssd0/ucinlp/shivag5/tmp/ray')
    search_args = dict(
        search=True,
        skip_memory_metrics=True,   # without this trainer.hyperparameter_search() + raytune bugs out
        disable_tqdm=True
    )
    if task_id == 1:
        from task1.params import get_args, get_hp_space
        args = get_args(cmd_args=cmd_args, **search_args)
        metric = MetricParams(name='rmse', direction='minimize')
    else:
        model_id = int(cmd_args.hp_space.split('_')[0][-1:])
        from task2.params import get_args, get_hp_space
        args = get_args(cmd_args=cmd_args, model_id=model_id, **search_args)
        metric = MetricParams(name='accuracy', direction='maximize')
    args.search_name = cmd_args.hp_space
    args.load_best_model_at_end = False
    args.save_strategy = IntervalStrategy.NO
    print(args)

    hp_space, is_grid = get_hp_space(cmd_args)
    print(hp_space(None), silent=silent)
    # import sys; sys.exit()

    scheduler = ASHAScheduler(
        max_t=args.num_train_epochs,
        grace_period=min(5, args.num_train_epochs),
        reduction_factor=2)

    search_alg = None
    if cmd_args.hyperopt:
        current_best_params = []
        print(f'Using hyperopt with best guesses: {current_best_params}', silent=silent)
        search_alg = HyperOptSearch(metric=f"eval_{metric.name}", mode=metric.mode,
                                    random_state_seed=args.seed,
                                    points_to_evaluate=current_best_params)
    if is_grid:
        cmd_args.num_samples = 1
    from ray.tune import CLIReporter

    analysis: ExperimentAnalysis = tune.run(
        tune.with_parameters(train, task_id=task_id, args=args, comet=cmd_args.comet),
        name=f'task-{task_id}/{args.search_name}',
        config=hp_space(None),
        local_dir=os.path.join(outputs_dir, 'ray_results'),
        metric=f'eval_{metric.name}',
        mode=metric.mode,
        num_samples=cmd_args.num_samples,
        resources_per_trial={"cpu": 2, 'gpu': cmd_args.gpus_per_trial},
        scheduler=scheduler,
        search_alg=search_alg,
        log_to_file=True,
        progress_reporter=CLIReporter(metric_columns=[f'eval_{metric.name}', 'epoch']),
        verbose=3,
        resume='ERRORED_ONLY'
    )
    print(analysis.best_result)
    print(f'Best hyperparameters found were: {analysis.best_config}')
    upload_confusion_matrices(task_id, args.search_name, search_key=str(analysis.trials[0]))

    if cmd_args.evaluate_best:
        print('Evaluating best model.')
        if task_id == 1:
            from task1.setup import setup
            args = get_args(cmd_args=cmd_args, search=True, **analysis.best_config)
        else:
            from task2.setup import setup
            args = get_args(cmd_args=cmd_args, model_id=model_id, search=True, **analysis.best_config)
        print(args)
        args.search_name = ''

        set_seed(args.seed)
        tokenizer, ds, model_init, trainer, metric = setup(args, data_dir=inputs_dir)
        if cmd_args.comet: trainer.add_callback(CometCallback)
        trainer.train()
        eval_metrics = trainer.evaluate()
        print(eval_metrics, silent=silent)

        if task_id == 2:    # confusion matrix in task 2's compute_metrics needs index_to_example_function
            from task2.setup import get_compute_metrics
            trainer.compute_metrics = get_compute_metrics(tokenizer=tokenizer,
                                                          ds=ds, split='test')

        test_metrics = trainer.evaluate(ds['test'], metric_key_prefix='test')
        print(test_metrics, silent=silent)


if __name__ == '__main__':
    parser = get_common_argparser()
    parser.add_argument('--task', type=int, choices=[1, 2], default=1)
    # parser.add_argument('--model', type=int, default=0, help='Model id for task 2')
    parser.add_argument('--hp_space', type=str, default='1')
    parser.add_argument('--hyperopt', action='store_true')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--gpus_per_trial', type=int, default=0)
    parser.add_argument('--evaluate_best', action='store_true')
    cmd_args = parser.parse_args()
    silent = cmd_args.silent
    print(cmd_args, silent=silent)

    main(cmd_args)

# cmd_args.comet=True
# cmd_args.hp_space='model1_base'
# cmd_args.task=2
# cmd_args.overwrite=True
# cmd_args.gpus_per_trial=1
# cmd_args.num_epochs=2
