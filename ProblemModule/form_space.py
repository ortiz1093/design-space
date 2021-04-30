import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


class FormSpace:
    def __init__(self):
        self.symbols = None
        self.solution_points = None
        self.fail_points = None
        self.points_df = None

    def input_data(self, points_dict, solution_flags):
        self.points_df = pd.DataFrame(points_dict)
        self.solution_points = solution_flags

        cats = self.points_df.select_dtypes(exclude=[int, float]).astype('category')
        for col in cats.columns:
            codes = cats[col].cat.codes
            self.points_df[col] = codes

        pass

    def dep_input_data(self, points_dict, solution_flags):
        self.symbols = list(points_dict.keys())
        points = np.vstack(tuple(points_dict.values()))

        self.solution_points = points[:, solution_flags].T
        self.fail_points = points[:, ~solution_flags].T

    def _1_show_solution_space(self, max_dim=15, show_fails=True):
        """
        Produces a matrix of scatter plots to allow multiple plots to share an
        axis for each basis dimension in the diffusion maps. Produces one
        subplot per pair of dimensions up to the smaller number of dimensions
        between the two datasets or a specified maximum, whichever is less.
        Displays automatically.

        parameters:
            data1: |samples x dimensions| numpy array for dataset 1 d-maps
            data2: |samples x dimensions| numpy array for dataset 2 d-maps
            labels [optional]: labels to identify data1 and data2 in the plots
            max_dim [optional]: max number of dimensions to plot

        returns:
            Nothing
        """

        data1 = self.solution_points
        data2 = self.fail_points

        labels = ('Pass', 'Fail')

        n_dim = min(max_dim, data1.shape[1], data2.shape[1])
        idx = list(range(n_dim))

        marker_size = 5
        marker_opacity = 1.0
        show_legend = True

        trace1 = go.Splom(
            dimensions=[dict(label=self.symbols[i],
                             values=data1[:, i])
                        for i in idx],
            # diagonal_visible=False,
            # showupperhalf=False,
            # opacity=marker_opacity,
            # marker=dict(color=0,
            #             size=marker_size,
            #             colorscale='Bluered',
            #             line=dict(width=0.5,
            #                       color='rgb(230,230,230)')
            #             ),
            text=labels[0],
            name=labels[0],
            # diagonal=dict(visible=False),
            # showlegend=show_legend
        )

        trace2 = go.Splom(
            dimensions=[dict(label=self.symbols[i], values=data2[:, i])
                        for i in idx] if show_fails else [],
            # diagonal_visible=False,
            # showupperhalf=False,
            # opacity=marker_opacity,
            # marker=dict(color=1,
            #             size=marker_size,
            #             colorscale='Bluered',
            #             line=dict(width=0.5,
            #                       color='rgb(230,230,230)')
            #             ),
            text=labels[1],
            name=labels[1],
            # diagonal=dict(visible=False),
            # showlegend=show_legend
        )

        fig = go.Figure(data=[trace1, trace2])
        fig.update_traces(
            showupperhalf=False,
            opacity=marker_opacity,
            marker=dict(color=1,
                        size=marker_size,
                        colorscale='Bluered',
                        line=dict(width=0.5,
                                  color='rgb(230,230,230)')
                        ),
            diagonal=dict(visible=False),
            showlegend=show_legend
        )
        fig.update_layout(
            legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=0.8,
                    xanchor="right",
                    x=0.7
            )
        )
        fig.update_xaxes(overwrite=True, tickformat=".3s")

        # # suffix = [1,2,3,4]
        # for i in range(1, n_dim):
        #     fig.update_layout(**{f"xaxis{i}": dict(tickformat="%"),
        #                          f"yaxis{i+1}": dict(tickformat="%")
        #                          })
        #     # fig.update_layout(**{f"xaxis{i}": dict(showticklabels=False),
        #     #                      f"yaxis{i}": dict(showticklabels=False)
        #     #                      })

        fig.show()  # opens new tab/window in default browser

    def show_solution_space(self, max_dim=10, show_fails=True):
        df = self.points_df.astype(float)
        # df = df.astype(float)

        hue = None
        if show_fails:
            hue = 'solution'
            df[hue] = self.solution_points
        else:
            df = df[self.solution_points]


        g = sns.pairplot(df, hue=hue, diag_kind='kde', corner=True, aspect=1, height=1)
        g.map_lower(sns.kdeplot, levels=4, color=".2")
        # plt.show()

    def _3_show_solution_space(self, max_dim=10, show_fails=True):
        marker_size = 5
        marker_opacity = 1.0
        show_legend = True

        df = pd.DataFrame(self.fail_points, columns=self.symbols)
        fig = px.scatter_matrix(df)
        # fig.update_traces(diagonal_visible=False)
        fig.update_traces(
            showupperhalf=False,
            opacity=marker_opacity,
            marker=dict(color=1,
                        size=marker_size,
                        colorscale='Bluered',
                        line=dict(width=0.5,
                                  color='rgb(230,230,230)')
                        ),
            diagonal=dict(visible=False),
            showlegend=show_legend
        )
        fig.update_xaxes(overwrite=True, tickformat=".3s")
        fig.show()
