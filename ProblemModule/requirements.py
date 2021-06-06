import numpy as np
import json


class Requirement:
    # TODO: Add capability for non-interval values (low priority)
    def __init__(self, symbol, allowable_vals, text):
        self.symbol = symbol
        self.values = allowable_vals
        self.text = text

    def __repr__(self):
        return f'Requirement({self.symbol}, {self.values}, {self.text})'

    def __str__(self):
        return self.text

    def check_compliance(self, test_val):
        # TODO: Implement model-based compliance checking (hi priority)
        if test_val.dtype == 'bool':
            return test_val

        return np.logical_and(np.min(self.values) <= test_val,
                              test_val <= np.max(self.values))


class RequirementSet:
    # TODO: Method for modifying allowable values (extremely low priority)
    # TODO: Method for modifying symbols (extremely low priority)
    # TODO: Method for deleting requirements (extremely low priority)
    def __init__(self):
        self.requirements = None

    def __getitem__(self, symbol):
        if isinstance(symbol, int):
            return self.requirements[symbol]
        return next((req for req in self.requirements
                     if req.symbol == symbol), None)

    def __len__(self):
        return 0 if self.requirements is None else len(self.requirements)

    def append_requirements(self, symbols, allowable_values, text):
        if self.requirements is None:
            self.requirements = []

        for symbol, values, text in zip(symbols, allowable_values, text):
            if np.max(values) >= 1e300:
                max_idx = values.index(np.max(values))
                values[max_idx] = np.inf
            self.requirements.append(Requirement(symbol, values, text))

    def append_requirements_from_json(self, filepath):
        with open(filepath, "r") as f:
            reqs = json.load(f)

        symbols = list(reqs.keys())
        values = [item['allowable range'] for item in reqs.values()]
        text = [item['text'] for item in reqs.values()]

        self.append_requirements(symbols, values, text)

    def check_compliance(self, points_dict):
        pass_ = np.all(
            np.vstack(
                tuple(self[sym].check_compliance(val)
                      for sym, val in points_dict.items())
                ),
            0)

        return pass_


if __name__ == "__main__":
    def test():
        # test_val = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        test_file = "reqs.json"
        test_points = {'a1': np.array([0.01, 0.2, 0.9, 0.6]),
                       'a2': np.array([0.5, 2, 3, 1.1]),
                       'a3': np.array([0, 0.3, 5.6, 1.1]),
                       'a4': np.array([0, 11.2, 75.01, 62])}

        reqs = RequirementSet()
        reqs.append_requirements_from_json(test_file)
        reqs.check_compliance(test_points)

        pass

    test()
