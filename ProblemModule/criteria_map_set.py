from map_set import MapFunction
from sympy import sympify, lambdify
import numpy as np
import json


class CriteriaMap:
    def __init__(self):
        self.functions = None
        self.inputs = None
    
    def update_inputs(self):
        if self.inputs is None:
            self.inputs = set()

        self.inputs.update([i for f in self.functions for i in f.inputs])

    def append_functions(self, symbols, expressions):
        if self.functions is None:
            self.functions = []

        for symbol, expr in zip(symbols, expressions):
            self.functions.append(MapFunction(symbol, expr))
        
        self.update_inputs()

    def append_functions_from_json(self, filepath):
        with open(filepath, "r") as f:
            funcs = json.load(f)

        symbols = list(funcs.keys())
        exprs = list(funcs.values())

        self.append_functions(symbols, exprs)

    def map_points(self, points_dict):
        criteria_pts = dict()
        for fun in self.functions:
            fun_args = {str(key): points_dict[str(key)] for key in fun.inputs}
            criteria_pts[fun.symbol] = fun.eval(fun_args)

        return criteria_pts


if __name__ == "__main__":
    from numpy.random import default_rng

    rng = default_rng(42)

    def test():
        filepath = "3DP_criteria.json"

        points = {
            'mu': rng.random(30) * (0.25 - 0.0625) + 0.0625,
            'p': rng.random(30) * 2 + 1,
            'n': rng.integers(15, 50, 30) * 50,
            'phi': rng.random(30) * 1 + 1,
            'V': rng.random(30) * 21 + 3,
            'Amp': rng.random(30) * 3,
            's': rng.random(30) * 6 + 21,
        }

        test_map = CriteriaMap()
        test_map.append_functions_from_json(filepath)

        pts = test_map.map_points(points)

        return pts

    constraint_pts = test()
    pass
