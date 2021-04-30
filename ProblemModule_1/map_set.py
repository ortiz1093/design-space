# import sympy as sym
# import numpy as np


class MapFunction:
    def __init__(self, symbol, expression):
        self.symbol = symbol
        self.expr = expression


class Map:
    def __init__(self):
        self.functions = None
        self.symbols = None

    def append_functions_from_file(self, file):
        pass
