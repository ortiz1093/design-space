import plotly.graph_objects as go
import numpy as np
from numpy.random import default_rng

rng = default_rng(1230)

def choose_ndim_points(axis_intervals, N):
    intervals = np.nan_to_num(axis_intervals)
    return np.array(
        [rng.uniform(*interval, N) for interval in intervals]
    )

N = 50000
pts = choose_ndim_points([[5, 30], [1, 10], [-2, 2]], N)
x = choose_ndim_points([[5, 30], [15, 20], [-2, 2]], 1).flatten()

dists = np.linalg.norm(pts.T - np.tile(x, [N, 1]), axis=1)
NN_idx = np.argmin(dists)
NN = pts.T[NN_idx]

NN_line = list(zip(x, NN))
fig = go.Figure(
    data=[
        go.Scatter3d(
            x=pts[0], y=pts[1], z=pts[2],
            mode='markers', marker_size=5,
            marker_line_color='DarkSlateGrey', marker_line_width=2,
            name='Solution Space'),
        go.Scatter3d(
            x=[x[0]], y=[x[1]], z=[x[2]],
            mode='markers', marker_size=8, marker_color='green',
            marker_line_color='DarkSlateGrey', marker_line_width=2,
            name='Desired Point'),
        go.Scatter3d(
            x=[NN[0]], y=[NN[1]], z=[NN[2]],
            mode='markers', marker_size=8, marker_color='red',
            marker_line_color='DarkSlateGrey', marker_line_width=2,
            name='Nearest Neigbor'),
        go.Scatter3d(
            x=NN_line[0], y=NN_line[1], z=NN_line[2],
            mode='lines', line_color='black', line_width=5,
            name='Shortest distance')
    ],
    layout_scene_aspectmode='data',
    layout_scene_aspectratio=dict(x=1, y=1, z=0.95)
)

fig.show()
