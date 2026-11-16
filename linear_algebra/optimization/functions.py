import numpy as np

class BenchmarkFunctions:
    """Library of standard optimization test functions."""
    
    @staticmethod
    def sphere(x):
        val = np.sum(x**2)
        grad = 2 * x
        hess = 2 * np.eye(len(x))
        return val, grad, hess

    @staticmethod
    def rosenbrock(x):
        """Rosenbrock function: f(x, y) = (1-x)^2 + 100(y-x^2)^2"""
        if len(x) != 2:
            raise ValueError("Rosenbrock defined here for 2D only.")
        x1, x2 = x[0], x[1]
        val = (1 - x1)**2 + 100 * (x2 - x1**2)**2
        grad = np.array([
            -2 * (1 - x1) - 400 * x1 * (x2 - x1**2),
            200 * (x2 - x1**2)
        ])
        hess = np.array([
            [2 - 400 * x2 + 1200 * x1**2, -400 * x1],
            [-400 * x1, 200]
        ])
        return val, grad, hess

    @staticmethod
    def himmelblau(x):
        """Himmelblau's function: f(x, y) = (x^2+y-11)^2 + (x+y^2-7)^2"""
        if len(x) != 2:
            raise ValueError("Himmelblau defined for 2D only.")
        x1, x2 = x[0], x[1]
        val = (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2
        grad = np.array([
            4 * x1 * (x1**2 + x2 - 11) + 2 * (x1 + x2**2 - 7),
            2 * (x1**2 + x2 - 11) + 4 * x2 * (x1 + x2**2 - 7)
        ])
        hess = np.array([
            [12 * x1**2 + 4 * x2 - 42, 4 * x1 + 4 * x2],
            [4 * x1 + 4 * x2, 4 * x1 + 12 * x2**2 - 26]
        ])
        return val, grad, hess
