import numpy as np
from pprint import pprint
import json


def _generate_conitnuous_variable_points(rg, num_pts=1):
    if num_pts:
        return np.random.random_sample(num_pts)*np.diff(rg) + np.min(rg)
    else:
        return float(np.random.random_sample(num_pts)*np.diff(rg) + np.min(rg))


def _generate_discrete_variable_points(rg, num_pts=1):
    rg[1] += 1

    return np.random.choice(range(*rg), num_pts)


def _generate_explicit_variable_points(vals, num_pts=1):
    return np.random.choice(vals, num_pts)


def _generate_coupled_variable_points(coupled_variable_values, num_pts=1):
    options = list(coupled_variable_values.keys())
    choices = np.random.choice(options, num_pts)

    syms = [sym for choice in choices
            for sym in coupled_variable_values[choice].keys()]
    vals = [val for choice in choices
            for val in coupled_variable_values[choice].values()]

    D = dict()
    _ = list(map(lambda x, y: D.setdefault(x, []).append(y), syms, vals))

    return [(sym, np.array(vals)) for sym, vals in D.items()]


def form_space_point_generator(design_var_dict, N=1):
    actions = {
        'continuous': _generate_conitnuous_variable_points,
        'discrete': _generate_discrete_variable_points,
        'explicit': _generate_explicit_variable_points,
        'coupled': _generate_coupled_variable_points
    }

    points = []

    for sym, info in design_var_dict.items():
        _, type_, vals = info.values()
        pt_val = actions[type_](vals, num_pts=N)
        points.extend(pt_val if type_ == 'coupled' else [(sym, pt_val)])
        pass

    return dict(points)


if __name__ == "__main__":

    with open("design_vars.json", "r") as f:
        D = json.load(f)

    point_dict = form_space_point_generator(D, N=5)

    pprint(point_dict)

    pass
