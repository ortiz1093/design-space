# from datetime import datetime
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import sympy as sym
import numpy as np
from itertools import chain
from Requirements import RequirementSet
from constraint_space import continuous_continuous_space


class DesignVariable:
    def __init__(self, description=None, symbol=None, space_type=None,
                 search_space=None):

        def __get_description(self, description=None):
            if description:
                assert isinstance(description, str), \
                    "variable description must be string"
                self.description = description

        assert isinstance(symbol, str), \
            "variable symbol must be string"

        assert space_type in ['continuous', 'discrete',
                              'explicit', 'coupled'], \
            "space_type must be either 'continuous', 'discrete', 'explicit'," \
            " or 'coupled'"

        assert type(search_space) in ['list'], \
            "search_space must be of type 'list' or 'set'"

        self.description = description
        self.symbol = symbol
        self.space_type = space_type
        self.search_space = search_space


class Problem:
    # TODO: Add design variables of different types
    # TODO: Create solution space
    # TODO: Migrate "eval_points" func from batch_funcs
    def __init__(self):
        self.A = []
        self.B = []
        self.M = dict()
        self.P = None
        self.S = None
        self.Omega = None
        self.R = RequirementSet()

    def add_requirement(self, text=None, symbol=None, values=None, file=None):
        if file:
            assert all([not text, not symbol, not values]), \
                "Cannot provide file and individual requirement"
            self.R._batch_add(file)
            self.__generate_param_set()
        else:
            self.R._add_requirement(text=text, symbol=symbol,
                                    values=values)

        self.__generate_param_set()

    def get_requirements_list(self):
        return [req for req in self.R]

    def __generate_param_set(self):
        cursor = self.R._RequirementSet__head
        flag = True
        while flag:
            flag = True if cursor._Requirement__next else False
            if cursor.symbol not in self.A:
                self.A.append(cursor.symbol)
            else:
                pass
            cursor = cursor._Requirement__next

        assert len(self.R) == len(set(self.A)), "Number of requirements " \
            "doesn't match number of symbols. Possible duplicate."

    def show_param_set(self):
        print("A = ", self.A)

    def _generate_problem_space(self, pairwise=True, color='plum'):
        N_axes = len(self.A)
        probFig = make_subplots(rows=N_axes-1, cols=N_axes-1)
        if pairwise:
            for i in range(1, N_axes):
                for ii in range(i, N_axes):
                    ReqA = self.R[ii].values
                    ReqB = self.R[i-1].values
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
                        probFig.update_yaxes(title_text=self.R[i-1].symbol,
                                             row=i, col=ii)
                        probFig.update_xaxes(title_text=self.R[ii].symbol,
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

    def _map_from_file(self, file):
        with open(file, "r") as f:
            lines = f.readlines()

        idx = list(range(len(lines)))[::3]

        for i in idx:
            req = lines[i].replace("\n", " ").strip()
            expr = lines[i+1].replace("\n", " ").strip()
            self.add_map_function(req, sym.sympify(expr))

        self._set_design_vars()
        self._report_missing_functions()

    def add_map_function(self, symbol, expression):
        self.M.update({symbol: expression})

    def _report_missing_functions(self):
        missing = set(self.A)-set(self.M)
        if missing:
            print('The following constraint parameters do not have map '
                  f'functions: {missing}')

        extra = set(self.M)-set(self.A)
        if extra:
            print('The following functions are not associated with existing '
                  f'constraints: {extra}')

    def _set_design_vars(self):

        syms = np.array([expr.free_symbols for expr in self.M.values()]) \
            .flatten()
        self.B = list(set(chain.from_iterable(syms)))

    def _eval_points(self):
        # TODO: Modify to work in Problem class
        results = np.array([]).reshape(0, points['n'])
        for expr in exprs:
            syms = tuple(_expected_vars(expr))
            f = sym.lambdify(syms, expr, "numpy")
            P = {str(k): D[str(k)] for k in syms}
            results = np.vstack((results, f(**P)))

        return results

    def _design_variables_from_file(self, file):
        with open(file, 'r') as f:
            vars_ = json.load(f)

        pass

    def _check_map(self):
        # TODO: Check design variables against map
        # TODO: How to distinguish function inputs from design vars in code
        # (i.e. "motor" from "voltage" or "inductance")
        pass

    def _calculate_sample_space(self):
        # TODO: Determine the sample space from design variables
        pass

    def _generate_solution_space(self, pairwise=True, color='ForestGreen',
                                 show_fails=False):
        # TODO: Determine the solution space from the sample space and reqs
        pass


if __name__ == "__main__":
    printer = Problem()
    printer.add_requirement(file="reqs.txt")
    printer._map_from_file("test_funcs.txt")
    printer._generate_problem_space()

    pass
