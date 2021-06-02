import os
import sys
import argparse
import comet_ml

# Better to set CUDA_VISIBLEDEVICES while running the script as: `CUDA_VISIBLE_DEVICES=0 python task1-driver.py <options>``
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['COMET_PROJECT_NAME'] = 'humor-1'
os.environ['COMET_MODE'] = 'ONLINE'

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(root_folder)
sys.path.insert(0, root_folder)

from task1.util import setup
from task1.params import Task1Arguments, get_args
from util import get_common_argparser, print_ds_stats, output as print

parser = get_common_argparser()
parser.add_argument("--transformer", type=str, default='bert-base-cased')
parser.add_argument("--freeze_transformer", action="store_true")
parser.add_argument("--add_word_embs", action='store_true')
parser.add_argument("--add_amb_embs", action='store_true')
cmd_args = parser.parse_args()
print(cmd_args, silent=cmd_args.silent)

args: Task1Arguments = get_args(cmd_args=cmd_args)

tokenizer, ds, model, trainer = setup(args)
print_ds_stats(ds, silent=cmd_args.silent)

trainer.train()
trainer.evaluate()
