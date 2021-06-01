import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from mca import categorize1D, factor_space_projection


class GaugeSpace:
    def __init__(self):
        self.symbols = None
        self.points_df = None
        self.plot = None
        self.cost_func = None

    def set_value_gradient(self, gradient_values):
        self.gradient = gradient_values

    # def input_data(self, points_dict):
    #     self.points_df = pd.DataFrame(points_dict)

    #     self.categorical_axes = self.points_df \
    #         .select_dtypes(exclude=[int, float]) \
    #         .astype('category')

    #     self.numerical_axes = self.points_df \
    #         .select_dtypes(include=[int, float])

    #     for col in self.categorical_axes.columns:
    #         codes = self.categorical_axes[col].cat.codes
    #         self.points_df[col] = codes

    def input_data(self, points_dict):
        self.points_df = pd.DataFrame(points_dict)

    def build_gauge_space(self, max_dim=10, show_gradient=False):
        df = self.points_df.astype(float)

        if df.shape[1] > max_dim:
            df = df.iloc[:, :max_dim]

        hue = None
        if show_gradient:
            hue = 'gradient'
            # hue = self.gradient[self.solution_points]
            df[hue] = self.gradient

        diag_kind = 'auto' if show_gradient else 'kde'

        self.plot = sns.pairplot(df, hue=hue, diag_kind=diag_kind,
                                 corner=True, aspect=1, height=1)

        if not show_gradient:
            self.plot.map_lower(sns.kdeplot, levels=4, color=".2")

    def show_gauge_space(self, **kwargs):
        if self.plot is None:
            self.build_gauge_space(**kwargs)

        plt.show()
