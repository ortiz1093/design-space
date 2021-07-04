import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ProblemModule.utils import continuous_continuous_space, dep_continuous_continuous_space, problem_space_grid

go.Figure.problem_space_grid = problem_space_grid


class ConstraintSpace:
    # TODO: Method for individual pair plots (low priority)
    # TODO: Method for plotting 3 requirements (low priority)
    def __init__(self):
        self.requirement_set = None
        self.figure = None

    def set_requirements(self, requirement_set):
        self.requirement_set = requirement_set

    def dep_build_problem_space(self, color='plum'):
        # TODO: Migrate constraint space plot to seaborn (low priority)
        N_axes = len(self.requirement_set)
        self.figure = make_subplots(rows=N_axes-1, cols=N_axes-1)
        for i in range(1, N_axes):
            for ii in range(i, N_axes):
                ReqA = self.requirement_set[ii].values
                ReqB = self.requirement_set[i-1].values
                x0, x1 = min(ReqA), min(max(ReqA), 1e300)
                y0, y1 = min(ReqB), min(max(ReqB), 1e300)
                self.figure.add_trace(
                    go.Scatter(x=[x0, x1, x1, x0, x0], y=[y0, y0, y1, y1, y0],
                               showlegend=True if i + ii == 2 else False
                               ),
                    row=i, col=ii
                )

                if i == ii:
                    self.figure.update_yaxes(
                        title_text=self.requirement_set[i-1].symbol,
                        row=i, col=ii)
                    self.figure.update_xaxes(
                        title_text=self.requirement_set[ii].symbol,
                        row=i, col=ii)

        self.figure.update_traces(
            mode='lines',
            fill='tonexty',
            fillcolor='plum',
            line=dict(color='plum'),
            name='Problem Space',
            legendgroup="group"
        )
        # self.figure.update_xaxes(range=[])
        self.figure.update_layout(title="Constraint Space -- Pairwise Axes",
                                  showlegend=True,
                                  legend=dict(
                                      yanchor="bottom",
                                      y=1.0,
                                      xanchor="right",
                                      x=1.0
                                  ))

    def build_problem_space(self, color='plum'):
        assert self.requirement_set is not None, "No requirements specified. Cannot build problem space."
        
        ranges = [req.values for req in self.requirement_set]
        axis_labels = [req.symbol for req in self.requirement_set]
        self.figure = go.Figure()

        self.figure.problem_space_grid(ranges, axis_labels=axis_labels, fillcolor=color, opacity=0.8)

    def show_problem_space(self):
        if self.figure is None:
            self.build_problem_space()

        self.figure.show()
