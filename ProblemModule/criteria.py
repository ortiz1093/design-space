import numpy as np
import json


class Criterion:
    # TODO: Add capability for non-interval values (low priority)
    def __init__(self, symbol, weight, text):
        raise NotImplementedError("Criterion class under construction")
        self.symbol = symbol
        self.weight = weight
        self.text = text

    def __repr__(self):
        return f'Criterion({self.symbol}, {self.values}, {self.text})'

    def __str__(self):
        return self.text


class CriteriaSet:
    def __init__(self):
        raise NotImplementedError("CriteriaSet class under construction")
        self.criteria = None

    def __getitem__(self, symbol):
        if isinstance(symbol, int):
            return self.requirements[symbol]
        return next((req for req in self.requirements
                     if req.symbol == symbol), None)

    def __len__(self):
        return 0 if self.requirements is None else len(self.requirements)

    def append_criteria(self, symbols, weights, text):
        if self.criteria is None:
            self.criteria = []

        for symbol, weights, text in zip(symbols, weights, text):
            # TODO: Process params before appending
            self.criteria.append(Criterion(symbol, weights, text))

    def append_criteria_from_json(self, filepath):
        with open(filepath, "r") as f:
            crit = json.load(f)

        symbols = list(crit.keys())
        weights = [item['weight'] for item in reqs.values()]
        text = [item['text'] for item in reqs.values()]

        self.append_criteria(symbols, weights, text)


if __name__ == "__main__":
    def test():
        pass

    test()
