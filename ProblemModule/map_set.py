from sympy import sympify, lambdify
import numpy as np
import json
import dill


class MapModel:
    # TODO: Switch to regression models and merge with MapFunction for consistency
    def __init__(self, symbol, inputs, model, X_mins, X_maxs):
        self.symbol = symbol
        self.model = model
        self.inputs = inputs
        self.x_lims = (X_mins, X_maxs)

    def args_dict_2_numpy(self, args_dict):
        assert np.all([input_ in args_dict.keys() for input_ in self.inputs]), "Missing input variables"
        
        return np.array([args_dict[sym] for sym in self.inputs]).T

    def scale_points(self, points):
        mins = self.x_lims[0]
        maxs = self.x_lims[1]
        return (points - mins) / (maxs - mins)

    def predict(self, points):
        pts_array = self.args_dict_2_numpy(points)
        scaled_pts = self.scale_points(pts_array)
        return self.model.predict(scaled_pts)

        # return f(**points)


class MapFunction:
    # TODO: Incorporate quantities and constants for use in mapping (hi priority). These are
    #       not design variables or constraint parameters, just values used to
    #       calculate them. i.e. gravity, intermediate dependent variables, etc
    # TODO: Method for modifying existing functions (low priority)
    # TODO: Method for removing map functions from set (low priority)
    def __init__(self, symbol, str_expression):
        self.symbol = symbol
        self.expression = sympify(str_expression)
        self.inputs = list(self.expression.free_symbols)

    def eval(self, points):
        # TODO: Implement lookups between map_functions (hi priority)
        if self.expression:
            f = lambdify(self.inputs, self.expression, 'numpy')

        return f(**points)


class MapSet:
    def __init__(self):
        self.functions = None
        self.models = None

    def append_functions(self, symbols, expressions):
        if self.functions is None:
            self.functions = []

        for symbol, expr in zip(symbols, expressions):
            self.functions.append(MapFunction(symbol, expr))

    def append_models(self, symbols, models, inputs_list):
        if self.models is None:
            self.models = []

        for symbol, model, inputs in zip(symbols, models, inputs_list):
            self.models.append(MapModel(symbol, inputs, *model))

    def load_model(self, model_path):
        with open(model_path, "rb") as f:
            model, mins, maxs = dill.load(f)
        
        return model, mins, maxs

    def append_functions_from_json(self, filepath):
        with open(filepath, "r") as f:
            map_dict = json.load(f)
        
        expr_syms = list(map_dict['exprs'].keys())
        exprs = list(map_dict['exprs'].values())

        model_syms = list(map_dict['models'].keys())
        models = [self.load_model(val[0]) for val in map_dict['models'].values()]
        model_inputs = [val[1] for val in map_dict['models'].values()]

        self.append_functions(expr_syms, exprs)
        self.append_models(model_syms, models, model_inputs)

    def map_points(self, points_dict):
        constraint_pts = {}
        for fun in self.functions:
            fun_args = {str(key): points_dict[str(key)] for key in fun.inputs}
            constraint_pts[fun.symbol] = fun.eval(fun_args)

        return constraint_pts
    
    def classify_points(self, points_dict):
        predictions = {}
        for model in self.models:
            fun_args = {str(key): points_dict[str(key)] for key in model.inputs}
            predictions[model.symbol] = model.predict(fun_args)

        return predictions


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
