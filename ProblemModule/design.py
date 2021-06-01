from map_set import MapSet
from design_variables import DesignVariable
from form_space import FormSpace
from requirements import RequirementSet
from constraint_space import ConstraintSpace
from gauge_space import GaugeSpace
from criteria_map_set import CriteriaMap
from value import ValueFunction
from warnings import warn
import matplotlib.pyplot as plt
import json
import dill


class Design:
    def __init__(self):
        self.design_variables = None
        self.map = MapSet()
        self.criteria_map = CriteriaMap()
        self.form_space = FormSpace()
        self.gauge_space = GaugeSpace()
        self.constraint_space = ConstraintSpace()
        self.constraint_points = None
        self.requirement_set = RequirementSet()
        self.map_inputs = None
        self.form_points = None
        self.gauge_points = None
        self.solution_flags = None
        self.value_function = None
        self.value_inputs = None
        self.gauge_inputs = None
        self.point_values = None
        self.criteria_points = None

    def set_value_function(self, str_expr):
        # TODO: Generate value function from criteria set (low-priority)
        self.value_function = ValueFunction(str_expr)

    def get_value_inputs(self):
        if self.value_inputs is None:
            self.value_inputs = {}

        inputs = self.value_function.inputs
        for input_ in inputs:
            var = str(input_)
            if var in self.map_inputs.keys():
                self.value_inputs[var] = self.map_inputs[var]
            elif var in self.constraint_points.keys():
                self.value_inputs[var] = self.constraint_points[var]
            else:
                raise NameError(f"symbol '{var}' is not defined in this design")

    def compute_point_values(self):
        assert self.value_function is not None, "No value function to evaluate"
        if self.value_inputs is None:
            self.get_value_inputs()

        self.point_values = self.value_function.eval(self.value_inputs)

    def load_requirements_from_json(self, filepath):
        self.requirement_set.append_requirements_from_json(filepath)
        self.constraint_space.set_requirements(self.requirement_set)

    def get_points(self, num_samples):
        samples = []
        for var in self.design_variables:
            samples.extend(var.generate_samples(num_samples))

        self.map_inputs = dict([(item[0], item[1])
                                for item in samples if item[2]])
        self.form_points = dict([(item[0], item[1])
                                 for item in samples if item[3]])

    def set_map_from_json(self, filepath):
        self.map.append_functions_from_json(filepath)

    def append_variables(self, symbols, sample_spaces):
        if self.design_variables is None:
            self.design_variables = []

        for symbol, space in zip(symbols, sample_spaces):
            self.design_variables.append(DesignVariable(symbol,
                                                        space[0],
                                                        space[1]))

        self.dimensions = len(self.design_variables)

    def append_variables_from_json(self, filepath):
        with open(filepath, "r") as f:
            vars_ = json.load(f)

        symbols = list(vars_.keys())
        spaces = list((val['values'], val['type']) for val in vars_.values())

        self.append_variables(symbols, spaces)

    def check_points(self, N=10, overwrite=False):
        if (self.map_inputs is None) or overwrite:
            self.get_points(N)
        self.constraint_points = self.map.map_points(self.map_inputs)
        self.solution_flags = self.requirement_set \
            .check_compliance(self.constraint_points)

    def build_constraint_space(self):
        self.constraint_space.build_problem_space()

    def build_form_space(self, N=10):
        self.check_points(N=N)
        self.form_space.input_data(self.form_points, self.solution_flags)

    def apply_value_gradient(self):
        if self.point_values is not None:
            self.form_space.set_value_gradient(self.point_values)
        else:
            warn("No gradient values to apply")

    def plot_problem(self):
        self.constraint_space.show_problem_space()

    def plot_solutions(self, **kwargs):
        self.form_space.show_solution_space(**kwargs)
        # self.form_space.pair_grid()

    def save(self, save_path):
        with open(save_path, "wb") as f:
            dill.dump(self, f)
    
    def set_criteria_map_from_json(self, filepath):
        self.criteria_map.append_functions_from_json(filepath)

    def get_criteria_inputs(self):
        self.criteria_points = {str(sym): self.map_inputs[str(sym)]
                             for sym in self.criteria_map.inputs
                             if str(sym) in self.map_inputs.keys()}

        self.criteria_points.update({str(sym): self.constraint_points[str(sym)]
                                  for sym in self.criteria_map.inputs
                                  if str(sym) in self.constraint_points.keys()})
        pass

    def build_gauge_space(self):
        if self.criteria_points is None:
            self.get_criteria_inputs()
        
        self.gauge_points = self.criteria_map.map_points(self.criteria_points)

        self.gauge_space.input_data(self.gauge_points)
    
    def plot_criteria(self):
        self.gauge_space.show_gauge_space()


if __name__ == "__main__":
    from time import time


    def normal():
        num_pts = 1_000
        t0 = time()

        reqs_file = '/root/ThesisCode/3DP_reqs.json'
        vars_file = '/root/ThesisCode/3DP_design_vars__new_motors.json'
        func_file = "/root/ThesisCode/3DP_funcs.json"

        print("#"*45)
        print("Initializing Design Object...")
        test_design = Design()

        print("Constructing Design Object from files...")
        print("\tRequirements...")
        test_design.load_requirements_from_json(reqs_file)
        print("\tDesign Variables...")
        test_design.append_variables_from_json(vars_file)
        print("\tMap...")
        test_design.set_map_from_json(func_file)
        print("Building Constraint Space...")
        test_design.build_constraint_space()
        test_design.plot_problem()
        print("Building Form Space...")
        test_design.build_form_space(N=num_pts)
        # test_design.plot_solutions(show_fails=True)
        test_design.plot_solutions(show_fails=False)

        print("#"*45)

        t = time() - t0
        print(f"\nDisplaying {num_pts} points")
        print(f"Elapsed time: {round(t, 2)}s")


    def reduction():
        num_pts = 1_000

        # Specify filenames
        reqs_file = '/root/ThesisCode/3DP_reqs.json'
        vars_file = '/root/ThesisCode/3DP_design_vars__new_motors.json'
        func_file = "/root/ThesisCode/3DP_funcs.json"

        # Create design from files
        test_design = Design()
        test_design.load_requirements_from_json(reqs_file)
        test_design.append_variables_from_json(vars_file)
        test_design.set_map_from_json(func_file)

        # Generate form space
        test_design.build_form_space(N=num_pts)
        test_design.plot_solutions(full_space=False, show_fails=False)


    def values():
        num_pts = 1_000

        # Specify filenames
        reqs_file = '/root/ThesisCode/3DP_reqs.json'
        vars_file = '/root/ThesisCode/3DP_design_vars__new_motors.json'
        func_file = "/root/ThesisCode/3DP_funcs.json"

        # Create design from files
        test_design = Design()
        test_design.load_requirements_from_json(reqs_file)
        test_design.append_variables_from_json(vars_file)
        test_design.set_map_from_json(func_file)

        # Generate form space
        test_design.build_form_space(N=num_pts)
        test_design.set_value_function('dx + q')
        test_design.compute_point_values()
        test_design.apply_value_gradient()
        # test_design.plot_solutions(full_space=True, show_fails=False)
        test_design.plot_solutions(full_space=True, show_fails=False, show_gradient=True)
        pass


    import seaborn as sns
    import pandas as pd
    import numpy as np
    import plom.plom_v4_4 as plom
    from plom.plom_utils import setupArgs
    import plotly.graph_objects as go


    def mixed_data_norms(X_uns, categorical_cols):
        cols = np.arange(X_uns.shape[1])
        cat_idx = [True if i in categorical_cols else False for i in cols]
        X = (X_uns - X_uns.min(0)) / (X_uns.max(0) - X_uns.min(0))
        X = X_uns

        norms = []
        for row in X:
            abs_diffs = np.abs(X - row)
            abs_diffs[:, cat_idx] = np.sign(abs_diffs[:, cat_idx])
            norms.append(np.sqrt(np.sum(abs_diffs**2, axis=1)))

        return np.array(norms)


    def mixed_data_dmap(X, categorical_cols, epsilon):
        norms = mixed_data_norms(X, categorical_cols)
        diffusions = np.exp(-norms / epsilon)
        scales = np.sum(diffusions, axis=0)**.5
        normalized = diffusions / (scales[:, None] * scales[None, :])
        values, vectors = np.linalg.eigh(normalized)
        basis_vectors = vectors / scales[:, None]
        basis = basis_vectors * values[None, :]

        return np.flip(basis,axis=1), np.flip(values)


    def reduction_dmaps():
        num_pts = 1_000

        # Specify filenames
        reqs_file = '/root/ThesisCode/3DP_reqs.json'
        vars_file = '/root/ThesisCode/3DP_design_vars__new_motors.json'
        func_file = "/root/ThesisCode/3DP_funcs.json"

        # Create design from files
        test_design = Design()
        test_design.load_requirements_from_json(reqs_file)
        test_design.append_variables_from_json(vars_file)
        test_design.set_map_from_json(func_file)

        # Generate form space
        test_design.build_form_space(N=num_pts)
        
        # Diffusion maps
        cat_cols = [8]
        df = test_design.form_space.points_df
        Y, _ = mixed_data_dmap(df.to_numpy(), cat_cols, 30)

        # Plotting seaborn
        new_df = pd.DataFrame(Y[:, :5])
        g = sns.pairplot(new_df, diag_kind='kde', corner=True, aspect=1, height=1)
        g.map_lower(sns.kdeplot, levels=4, color=".2")
        # plt.show()

        # Plotting plotly
        fig = go.Figure(data=[
            go.Scatter3d(x=Y[:, 1], y=Y[:, 2], z=Y[:, 3], mode='markers')
        ])
        fig.show()

        pass


    def gauge_test():
        num_pts = 1_000

        # Specify filenames
        reqs_file = '/root/ThesisCode/3DP_reqs.json'
        vars_file = '/root/ThesisCode/3DP_design_vars__new_motors.json'
        func_file = "/root/ThesisCode/3DP_funcs.json"
        criteria_file = "/root/ThesisCode/3DP_criteria.json"

        # Create design from files
        test_design = Design()
        test_design.load_requirements_from_json(reqs_file)
        test_design.append_variables_from_json(vars_file)
        test_design.set_map_from_json(func_file)
        test_design.set_criteria_map_from_json(criteria_file)

        # Generate form space
        test_design.build_form_space(N=num_pts)
        test_design.build_gauge_space()
        test_design.plot_criteria()

        # build guage space

        pass


    # values()
    # reduction()
    # normal()
    # reduction_dmaps()
    gauge_test()
    # plt.show()
    pass
