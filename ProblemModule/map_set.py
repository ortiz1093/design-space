from sympy import sympify, lambdify
import numpy as np
import json


class MapFunction:
    # TODO: Implement model-based mapping functionality (Hi priority)
    # TODO: Incorporate quantities and constants for use in mapping (hi priority). These are
    #       not design variables or constraint parameters, just values used to
    #       calculate them. i.e. gravity, intermediate dependent variables, etc
    # TODO: Method for modifying existing functions (low priority)
    # TODO: Method for removing map functions from set (low priority)
    def __init__(self, symbol, str_expression=None, model=None, model_inputs=None):
        assert not (str_expression and model), "Map function may have either a model or an expression, not both"
        if model:
            assert model_inputs is not None, "No inputs were provided for the model-based map"
        self.symbol = symbol
        self.expression = sympify(str_expression) if str_expression else None
        self.model = model
        self.inputs = list(self.expression.free_symbols)

    def args_dict_2_numpy(self, args_dict):
        # assert args_dict.keys() ==
        pass

    def eval(self, points):
        # TODO: Implement lookups between map_functions (hi priorities)
        if self.expression:
            f = lambdify(self.inputs, self.expression, 'numpy')
        else:
            # TODO: Implement eval for model-based mapping (Hi priority)
            pass

        return f(**points)


class MapSet:
    def __init__(self):
        self.functions = None

    def append_functions(self, symbols, expressions):
        if self.functions is None:
            self.functions = []

        for symbol, expr in zip(symbols, expressions):
            self.functions.append(MapFunction(symbol, expr))

    def append_functions_from_json(self, filepath):
        with open(filepath, "r") as f:
            funcs = json.load(f)

        symbols = list(funcs.keys())
        exprs = list(funcs.values())

        self.append_functions(symbols, exprs)

    def map_points(self, points_dict):
        constraint_pts = {}
        for fun in self.functions:
            fun_args = {str(key): points_dict[str(key)] for key in fun.inputs}
            constraint_pts[fun.symbol] = fun.eval(fun_args)

        return constraint_pts


if __name__ == "__main__":
    def test():
        filepath = "test_funcs.json"

        points = {
            'x': np.array([1, 2]),
            'y': np.array([3, 4]),
            'z': np.array([5, 6]),
            'p': np.array([7, 8]),
        }

        test_map = MapSet()
        test_map.append_functions_from_json(filepath)

        pts = test_map.map_points(points)

        return pts

    constraint_pts = test()
