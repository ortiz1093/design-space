import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


class FormSpace:
    def __init__(self):
        self.symbols = None
        self.solution_points = None
        self.fail_points = None
        self.points_df = None
        self.plot = None

    def input_data(self, points_dict, solution_flags):
        self.points_df = pd.DataFrame(points_dict)
        self.solution_points = solution_flags

        cats = self.points_df.select_dtypes(exclude=[int, float]) \
                   .astype('category')
        for col in cats.columns:
            codes = cats[col].cat.codes
            self.points_df[col] = codes

    def build_solution_space(self, max_dim=10, show_fails=False):
        df = self.points_df.astype(float)

        hue = None
        if show_fails:
            hue = 'solution'
            df[hue] = self.solution_points
        else:
            df = df[self.solution_points]

        self.plot = sns.pairplot(df, hue=hue, diag_kind='kde',
                                 corner=True, aspect=1, height=1)

        if not show_fails:
            self.plot.map_lower(sns.kdeplot, levels=4, color=".2")

    def show_solution_space(self, show_fails=False):
        if self.plot is None:
            self.build_solution_space(show_fails=show_fails)

        plt.show()
