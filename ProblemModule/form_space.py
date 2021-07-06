import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from ProblemModule.mca import categorize1D, factor_space_projection

import plotly.graph_objects as go
import plotly.figure_factory as ff


class FormSpace:
    # TODO: Implement feature extraction functionality (low priority)
    # TODO: Implement dimensionality reduction functionality (med priority)
    # TODO: Implement corr vals in plot matrix upper corner (low priority)
    # TODO: Implement ability to choose which variables to plot (low priority)
    # TODO: Implement 3D plotting capability (extremely low priority)
    # TODO: Implement MCA (or other mixed data dim-reduction) (med priority)
    def __init__(self):
        self.symbols = None
        self.solution_points = None
        # self.fail_points = None
        self.points_df = None
        self.plot = None
        self.reduced_df = None
        self.categorical_axes = None
        self.numerical_axes = None
        self.gradient = None

    def set_value_gradient(self, gradient_values):
        self.gradient = gradient_values

    def input_data(self, points_dict, solution_flags):
        self.points_df = pd.DataFrame(points_dict)
        self.solution_points = solution_flags

        self.categorical_axes = self.points_df \
            .select_dtypes(exclude=[int, float]) \
            .astype('category')

        self.numerical_axes = self.points_df \
            .select_dtypes(include=[int, float])

        for col in self.categorical_axes.columns:
            codes = self.categorical_axes[col].cat.codes
            self.points_df[col] = codes

    def build_solution_space(self, max_dim=10, full_space=True, show_fails=False, show_gradient=False, **kwargs):
        if full_space:
            df = self.points_df.astype(float)
        else:
            self.reduce_dims()
            df = self.reduced_df.astype(float)

        if df.shape[1] > max_dim:
            df = df.iloc[:, :max_dim]

        hue = None
        if show_fails:
            hue = 'solution'
            df[hue] = self.solution_points
        elif show_gradient:
            hue = 'utility'
            # hue = self.gradient[self.solution_points]
            df[hue] = self.gradient

        if full_space and not show_fails:
            df = df[self.solution_points]

        import plotly.express as px

        color_grad = df.utility.values
        diag_kind = 'auto' if show_gradient else 'kde'
        self.plot = px.scatter_matrix(
            df,
            dimensions=df.columns[:-1],
            color='utility')
        self.plot.update_traces(
            diagonal_visible=False,
            showupperhalf=False,
            marker=dict(
                size=4, opacity=1.0,
                showscale=False, # colors encode categorical variables
                line_color='whitesmoke', line_width=0.5))

        # self.plot = sns.pairplot(df, hue=hue, diag_kind=diag_kind,
        #                          corner=True, aspect=1, height=1)

        # if not (show_fails or show_gradient):
        #     self.plot.map_lower(sns.kdeplot, levels=4, color=".2")
        
        # def onclick(event):
        #     axes = event.inaxes
        #     axis_names = df.columns
        #     # x_name = axis_names[axes.colNum]
        #     x_name = axis_names[axes.get_subplotspec().colspan.start]
        #     # y_name = axis_names[axes.rowNum]
        #     y_name = axis_names[axes.get_subplotspec().rowspan.start]
        #     x = df[x_name]
        #     y = df[y_name]
        #     hu = df[hue] if hue else None
        #     plt.figure()
        #     clk_ax = sns.scatterplot(x=x, y=y, hue=hu)
        #     clk_ax.xaxis.xlabel = x_name
        #     clk_ax.xaxis.ylabel = y_name
        #     plt.show()
        #     # print(event.inaxes)
        #     # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #     #     ('double' if event.dblclick else 'single', event.button,
        #     #     event.x, event.y, event.xdata, event.ydata))

        # self.plot.fig.canvas.mpl_connect('button_press_event', onclick)

    def show_solution_space(self, **kwargs):
        if self.plot is None:
            self.build_solution_space(**kwargs)

        plt.show()
        # self.plot.show()

    def points2categorical(self):
        categorical_df = self.categorical_axes.copy(deep=True)

        for col in self.numerical_axes.columns:
            values = self.numerical_axes[col].to_numpy()
            categorical_df[col] = categorize1D(values)

        return categorical_df

    def reduce_dims(self):
        categorical_df = self.points2categorical()
        projection, contributions = factor_space_projection(categorical_df[self.solution_points])
        col_names = [f'Factor {i}' for i in range(len(contributions))]

        self.reduced_df = pd.DataFrame(projection, columns=col_names)

    def target_embedding(self):
        solns = pd.Series(self.solution_points, name="Solution")
        df = pd.concat([self.points_df, solns], axis=1)
        print(df)
        pass
