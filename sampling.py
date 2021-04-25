import numpy as np


def _stratified_sampling(low, high, divisions=10):

    divs = np.linspace(low, high, divisions+1)
    # print(divs)

    return np.random.rand(*divs[:-1].shape)*np.diff(divs) + divs[:-1]


def _sample_hypercube(axes, divisions=10):

    cube = np.array([], dtype=np.int64).reshape((0, divisions))
    for axis in axes:
        axis_samples = _stratified_sampling(*axis, divisions=divisions)
        np.random.shuffle(axis_samples)
        cube = np.vstack((cube, axis_samples))

    return cube


# def nested_hypercube(axes, axes_to_nest, min_max, divisions=10):
#
#     hypercube = _sample_hypercube(axes)


if __name__ == "__main__":
    Omega = {
        'gantry_length': [5.0, 30.00],
        'gantry_width': [0.36, 1.5],
        'gantry_height': [0.06, 0.75]
    }

    div = 15
    outer_axes = list(Omega.values())
    outer_cube = _sample_hypercube(outer_axes, divisions=div)

    sample_points = np.array([], dtype=np.int64).reshape(5, 0)
    for i in range(div):
        height = outer_cube[2, i]
        width = outer_cube[1, i]
        length = outer_cube[0, i]
        floor_thickness = [0.050, min(0.2, height-0.00001)]
        wall_thickness = [0.050, min(0.125, width/2-0.125, (length - 3)/6)]
        inner_axes = [floor_thickness, wall_thickness]
        inner_cube = _sample_hypercube(inner_axes, divisions=div)
        outer_params = np.tile(outer_cube[:, i], [div, 1]).T
        nested_cube = np.vstack((outer_params, inner_cube))
        sample_points = np.hstack((sample_points, nested_cube))

    with open('gantry_test_points.txt', 'w+') as f:
        print("length", "width", "height", "floor", "walls", file=f)
        for row in sample_points.T:
            print(*row.round(8), file=f, sep="\t")
