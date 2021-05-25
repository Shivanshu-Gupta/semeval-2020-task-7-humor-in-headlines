import os
import json
import attr
from typing import Union, Optional

from param_impl import default_value, Parameters

paths = json.load(open('paths.json'))
data_dir = paths['data_dir']
embeddings_dir = paths['embeddings_dir']
log_dir = paths['log_dir']
models_dir = paths['models_dir']
ray_dir = paths['ray_dir']

def disambiguate(o, t):
    lambdas = {
        Union[int, str]: lambda *_: None
    }
    if t in lambdas:
        return lambdas[t](o, t)
    else:
        raise TypeError("Unknown Type")
