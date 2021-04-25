import numpy as np
import plotly.graph_objects as go
from pprint import pprint


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


def _form_space_point_generator(design_var_dict, N=1):
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
    # x = {
    #     "A": {"L": 0.0015, "V": 1.3, "I": 0.2, "phi": 1.8},
    #     "B": {"L": 0.0032, "V": 5.0, "I": 1.0, "phi": 1.8},
    #     "C": {"L": 0.0020, "V": 1.6, "I": 0.4, "phi": 0.9}
    # }

    x = {
        "p": {
            "descr": "var to test continuous type",
            "type": "continuous",
            "values": [0.3, 0.9]
        },

        "x": {
            "descr": "var to test discrete type",
            "type": "discrete",
            "values": [10, 30, 1]
        },

        "y": {
            "descr": "var to test explicit type",
            "type": "explicit",
            "values": [1, 3, 15, 59, 101]
        },

        "z": {
            "descr": "var to test coupled type",
            "type": "coupled",
            "values": {
                "A": {"L": 0.0015, "V": 1.3, "I": 0.2, "phi": 1.8},
                "B": {"L": 0.0032, "V": 5.0, "I": 1.0, "phi": 1.8},
                "C": {"L": 0.0020, "V": 1.6, "I": 0.4, "phi": 0.9}
            }
        }
    }

    point_dict = _form_space_point_generator(x, N=5)

    pprint(point_dict)

    pass
