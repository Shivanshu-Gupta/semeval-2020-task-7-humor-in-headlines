# run task1: python driver.py <common options> task-1 <task 1 options>
# run task2: python driver.py <common options> task-2 model-<id> <task 2 model options>

import os
import comet_ml

from transformers import set_seed

import task1.params
import task2.params
from params import inputs_dir
from util import get_common_argparser, print_ds_stats, setup_comet, output as print

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def write_pred_file(task_id, tst_ds, preds, predfile=''):
    if not predfile: predfile = f'task-{task_id}-output.csv'
    from copy import deepcopy
    tst_ds = deepcopy(tst_ds)
    tst_ds._output_all_columns = True
    import pandas as pd
    results = pd.DataFrame(dict(id=tst_ds['id'], pred=preds))
    results.to_csv(predfile, index=False)

def main(cmd_args, task_id, args, model_id=None):
    print(args)
    set_seed(args.seed)
    tokenizer, ds, model_init, trainer, metric = setup(args, data_dir=inputs_dir)
    print_ds_stats(ds, silent=cmd_args.silent)

    if task_id != 2 or model_id != 0:   # Task 2 Model 0 simply compares grades predicted by a trained task 1 model
        trainer.train()

    # eval_metrics = trainer.evaluate()
    eval_preds, eval_labels, eval_metrics = trainer.predict(ds['validation'], metric_key_prefix='eval')
    print(eval_metrics)

    if task_id == 2:    # confusion matrix in task 2's compute_metrics needs index_to_example_function
        from task2.setup import get_compute_metrics
        trainer.compute_metrics = get_compute_metrics(tokenizer=tokenizer, ds=ds, split='test')

    test_preds, test_labels, test_metrics = trainer.predict(ds['test'], metric_key_prefix='test')
    print(test_metrics)

    if task_id == 2:    # task 2 models give logits
        eval_preds = eval_preds.argmax(-1) + 1
        test_preds = test_preds.argmax(-1) + 1

    if cmd_args.write_preds:
        from params import outputs_dir
        if task_id == 1:
            output_dir = os.path.join(outputs_dir, f'output/task-{task_id}/{args.model_name()}')
        else:
            output_dir = os.path.join(outputs_dir, f'output/task-{task_id}/model-{model_id}/{args.model_name()}')
        os.makedirs(output_dir, exist_ok=True)
        write_pred_file(task_id, ds['validation'], eval_preds, os.path.join(output_dir, 'dev-preds.csv'))
        write_pred_file(task_id, ds['test'], test_preds, os.path.join(output_dir, 'test-preds.csv'))

if __name__ == '__main__':
    parser = get_common_argparser()
    parser.add_argument('--gpu_idx', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.2)
    parser.add_argument('--write_preds', action='store_true')
    subparsers = parser.add_subparsers(help='Which task?', dest='task')

    # Parser for task 1
    parser_1 = subparsers.add_parser('task-1', help='Task 1 arguments')
    task1.params.setup_parser(parser_1)

    # Parser for task 2
    parser_2 = subparsers.add_parser('task-2', help='Task 2 arguments')
    task2.params.setup_parser(parser_2)

    cmd_args = parser.parse_args()
    print(cmd_args, silent=cmd_args.silent)

    task_id = int(cmd_args.task[-1:])

    if cmd_args.gpu_idx is not None:
        gpu = cmd_args.gpu_idx
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu) if gpu >= 0 else ''

    if cmd_args.comet: setup_comet(task_id=task_id)

    if task_id == 1:
        from task1.params import get_args
        from task1.setup import setup
        args = get_args(cmd_args=cmd_args)
        main(cmd_args, task_id, args)
    else:
        from task2.params import get_args
        from task2.setup import setup
        model_id = int(cmd_args.model[-1:])
        args = get_args(cmd_args=cmd_args, model_id=model_id)
        main(cmd_args, task_id, args, model_id=model_id)
