import numpy as np
from pprint import pprint


class DesignVariable:
    # TODO: Method for modifying symbol
    # TODO: Method for modifying search space
    # TOOD: Method for modifying subvariables
    def __init__(self, symbol, sample_space, space_type):
        self.symbol = symbol
        self.sample_space = sample_space
        self.space_type = space_type
        self.N_subvars = len(list(self.sample_space.values())[0]) \
            if space_type == 'coupled' else 0

    def _generate_conitnuous_variable_points(self, num_pts):
        map_flag = [True]*num_pts
        plot_flag = [True]*num_pts
        return list(zip([self.symbol],
                        [np.random.random_sample(num_pts)
                            * np.diff(self.sample_space)
                            + np.min(self.sample_space)],
                        map_flag,
                        plot_flag))

    def _generate_discrete_variable_points(self, num_pts):
        rng = self.sample_space
        rng[1] += 1

        map_flag = [True]*num_pts
        plot_flag = [True]*num_pts

        return list(zip([self.symbol],
                        [np.random.choice(range(*rng), num_pts)],
                        map_flag,
                        plot_flag))

    def _generate_explicit_variable_points(self, num_pts):
        map_flag = [True]*num_pts
        plot_flag = [True]*num_pts

        return list(zip([self.symbol],
                        [np.random.choice(self.sample_space, num_pts)],
                        map_flag,
                        plot_flag))

    def _generate_coupled_variable_points(self, num_pts):
        options = list(self.sample_space.keys())
        choices = np.random.choice(options, num_pts)

        map_flag = ([True]*self.N_subvars + [False])*num_pts
        plot_flag = ([False]*self.N_subvars + [True])*num_pts

        syms = [sym for choice in choices
                for sym in self.sample_space[choice].keys()]
        vals = [val for choice in choices
                for val in self.sample_space[choice].values()]

        D = dict()
        _ = list(map(lambda x, y: D.setdefault(x, []).append(y), syms, vals))
        D[self.symbol] = choices

        return [(sym, np.array(vals), map_flag[i], plot_flag[i])
                for i, (sym, vals) in enumerate(D.items())]

    def generate_samples(self, N):
        actions = {
            'continuous': self._generate_conitnuous_variable_points,
            'discrete': self._generate_discrete_variable_points,
            'explicit': self._generate_explicit_variable_points,
            'coupled': self._generate_coupled_variable_points
        }

        result = actions[self.space_type](num_pts=N)
        return result


if __name__ == "__main__":
    def test():
        p = DesignVariable('p', [0.3, 0.9], 'continuous')
        x = DesignVariable('x', [10, 30, 1], 'discrete')
        y = DesignVariable('y', [1, 3, 15, 59, 101], 'explicit')
        z = DesignVariable('z',
                           {
                               "A": {
                                   "L": 0.0015,
                                   "V": 1.3,
                                   "I": 0.2,
                                   "phi": 1.8},
                               "B": {
                                   "L": 0.0032,
                                   "V": 5.0,
                                   "I": 1.0,
                                   "phi": 1.8},
                               "C": {
                                   "L": 0.0020,
                                   "V": 1.6,
                                   "I": 0.4,
                                   "phi": 0.9}
                            },
                           'coupled')
        pprint(p.generate_samples(3))
        pprint(x.generate_samples(3))
        pprint(y.generate_samples(3))
        pprint(z.generate_samples(3))

    test()
