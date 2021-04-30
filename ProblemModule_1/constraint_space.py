import plotly.graph_objects as go


def continuous_continuous_space(rangeA, rangeB, color='plum', name=None,
                                showlegend=True):
    """
    Generate a plot showing a shaded continuous region of 2D space (rectangle).
        params:
            rangeA: upper and lower bounds of horizontal axis
            rangeB: upper and lower bounds of vertical axis
            color (optional): color of shaded region
            name (optional): name of shaded region to show in legend
            showlegend (optional): whether to represent trace in plot legend
        return:
            2D plotly scatter trace
    """
    x0, x1 = min(rangeA), max(rangeA)
    y0, y1 = min(rangeB), max(rangeB)

    return go.Scatter(
        x=[x0, x1, x1, x0, x0],
        y=[y0, y0, y1, y1, y0],
        fill='toself',
        fillcolor=color,
        line=dict(color=color),
        name=name,
        legendgroup="group",
        showlegend=showlegend
    )
