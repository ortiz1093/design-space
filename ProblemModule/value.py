from sympy import sympify, lambdify
import numpy as np
import json

class ValueFunction:
    def __init__(self, str_expression):
        # raise NotImplementedError("ValueFunction class under construction")
        self.expression = sympify(str_expression)
        self.inputs = list(self.expression.free_symbols)

    def eval(self, points):
        f = lambdify(self.inputs, self.expression, 'numpy')
        return f(**points)