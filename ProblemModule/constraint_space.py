import plotly.graph_objects as go
from plotly.subplots import make_subplots
# from requirements import RequirementSet


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
    x0, x1 = min(rangeA), min(max(rangeA), 1e300)
    y0, y1 = min(rangeB), min(max(rangeB), 1e300)

    return go.Scatter(
        x=[x0, x1, x1, x0, x0],
        y=[y0, y0, y1, y1, y0],
        mode='markers',
        fill='toself',
        fillcolor=color,
        line=dict(color=color),
        name=name,
        legendgroup="group",
        showlegend=showlegend
    )


class ConstraintSpace:
    def __init__(self):
        self.requirement_set = None

    def set_requirements(self, requirement_set):
        self.requirement_set = requirement_set

    def show_problem_space(self, color='plum'):
        N_axes = len(self.requirement_set)
        probFig = make_subplots(rows=N_axes-1, cols=N_axes-1)
        for i in range(1, N_axes):
            for ii in range(i, N_axes):
                ReqA = self.requirement_set[ii].values
                ReqB = self.requirement_set[i-1].values
                probFig.add_trace(
                    continuous_continuous_space(ReqA, ReqB,
                                                color=color,
                                                name='Problem Space',
                                                showlegend=True
                                                if i + ii == 2
                                                else False
                                                ),
                    row=i, col=ii
                )

                if i == ii:
                    probFig.update_yaxes(
                        title_text=self.requirement_set[i-1].symbol,
                        row=i, col=ii, )
                    probFig.update_xaxes(
                        title_text=self.requirement_set[ii].symbol,
                        row=i, col=ii)

        probFig.update_layout(title="Constraint Space -- Pairwise Axes",
                              showlegend=True,
                              legend=dict(
                                  yanchor="bottom",
                                  y=1.0,
                                  xanchor="right",
                                  x=1.0
                                ))
        probFig.show()
