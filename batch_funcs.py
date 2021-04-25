import numpy as np
import sympy as sym
from itertools import chain
import json

def _variables_from_file(file):
    with open(file, 'r') as f:
        vars_ = json.load(f)

    return vars_

v = _variables_from_file('design_vars.json')

pass