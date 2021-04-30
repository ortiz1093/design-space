import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def continuous_continuous_space(rangeA, rangeB):
    x0, x1 = min(rangeA), max(rangeA)
    y0, y1 = min(rangeB), max(rangeB)

    return go.Scatter(x=[x0, x1, x1, x0, x0],
                      y=[y0, y0, y1, y1, y0],
                      fill='toself')


R1 = np.array([0.3, 2.3])
R2 = np.array([0.6, 0.9])
R3 = np.array([3.6, 5.9])

testFig = make_subplots(rows=2, cols=2)
testFig.add_trace(continuous_continuous_space(R1, R2), row=1, col=1)
testFig.add_trace(continuous_continuous_space(R1, R3), row=1, col=2)
testFig.add_trace(continuous_continuous_space(R2, R3), row=2, col=2)

testFig.show()
