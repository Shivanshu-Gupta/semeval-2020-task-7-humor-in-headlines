# run task1: python driver.py <common options> task1 <task 1 options>
# run task2: python driver.py <common options> task2 <task 2 options>

import os
import comet_ml

from util import get_common_argparser, print_ds_stats, output as print

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['COMET_MODE'] = 'ONLINE'

parser = get_common_argparser()
parser.add_argument('--gpu_idx', type=int, default=None)
subparsers = parser.add_subparsers(help='Task arguments', dest='task')

# Parser for task 1
parser_1 = subparsers.add_parser('task1', help='Which task?')
parser_1.add_argument("--transformer", type=str, default='bert-base-cased')
parser_1.add_argument("--freeze_transformer", action="store_true")
parser_1.add_argument("--add_word_embs", action='store_true')
parser_1.add_argument("--add_amb_embs", action='store_true')

# Parser for task 2
parser_2 = subparsers.add_parser('task2', help='Task 2 arguments')
parser_2.add_argument("--transformer", type=str, default='bert-base-cased')
parser_2.add_argument("--combined", action='store_true')

cmd_args = parser.parse_args()
print(cmd_args, silent=cmd_args.silent)

if cmd_args.gpu_idx is not None:
    gpu = cmd_args.gpu_idx
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu) if gpu >= 0 else ''

task_id = int(cmd_args.task[-1:])
os.environ['COMET_PROJECT_NAME'] = f'humor-{task_id}'
if task_id == 1:
    from task1.params import Task1Arguments as TaskArguments, get_args
    from task1.setup import setup
else:
    from task2.params import Task2Arguments as TaskArguments, get_args
    from task2.setup import setup

args: TaskArguments = get_args(cmd_args=cmd_args)
tokenizer, ds, model, trainer, metric = setup(args)
print_ds_stats(ds, silent=cmd_args.silent)

trainer.train()
eval_metrics = trainer.evaluate()
print(eval_metrics)

if task_id == 2:    # confusion matrix in task 2's compute_metrics needs index_to_example_function
    from task2.setup import get_compute_metrics
    label_names = ds['train'].features['labels'].names
    trainer.compute_metrics = get_compute_metrics(tokenizer=tokenizer, ds=ds,
                                                    label_names=label_names, split='test')
test_metrics = trainer.evaluate(ds['test'], metric_key_prefix='test')
print(test_metrics)
