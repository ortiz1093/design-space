import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from mca import categorize1D, factor_space_projection


class FormSpace:
    # TODO: Implement feature extraction functionality (med priority)
    # TODO: Implement dimensionality reduction functionality (hi priority)
    # TODO: Implement corr vals in plot matrix upper corner (low priority)
    # TODO: Implement ability to choose which variables to plot (low priority)
    # TODO: Implement 3D plotting capability (extremely low priority)
    # TODO: Implement MCA (Hi priority)
    def __init__(self):
        self.symbols = None
        self.solution_points = None
        # self.fail_points = None
        self.points_df = None
        self.plot = None
        self.reduced_df = None
        self.categorical_axes = None
        self.numerical_axes = None

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

    def build_solution_space(self, max_dim=10, full_space=True, show_fails=False):
        if full_space:
            df = self.points_df.astype(float)
        else:
            self.reduce_dims()
            df = self.reduced_df.astype(float)

        if df.shape[1] > max_dim:
            df = df[:, :max_dim]

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

    def show_solution_space(self, **kwargs):
        if self.plot is None:
            self.build_solution_space(**kwargs)

        plt.show()

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
