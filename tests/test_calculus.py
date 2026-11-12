import pytest
import numpy as np
import sympy as sp
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.calculus.autodiff import Dual, AutoDiffEngine
from linear_algebra.calculus.numerical import FiniteDifferenceEngine
from linear_algebra.calculus.symbolic import SymbolicCalculusEngine
from linear_algebra.calculus.matrix_rules import MatrixCalculusIdentities
from linear_algebra.calculus.optimization import OptimizationEngine

def test_dual_number_autodiff():
    # f(x) = x^2 + sin(x)
    # df/dx = 2x + cos(x)
    def f(x):
        return x**2 + x.sin()
    
    # Eval at x = 3
    x_dual = Dual(3.0, 1.0)
    y_dual = f(x_dual)
    
    expected_real = 3.0**2 + np.sin(3.0)
    expected_dual = 2*(3.0) + np.cos(3.0)
    
    assert np.isclose(y_dual.real, expected_real)
    assert np.isclose(y_dual.dual, expected_dual)

def test_finite_difference_gradient():
    # f(x, y) = x^2 * y
    def f(x, y):
        return x**2 * y
        
    x0 = [2.0, 3.0]
    grad = FiniteDifferenceEngine.gradient(f, x0, h=1e-5)
    
    # grad = [2xy, x^2] -> at (2,3) = [12, 4]
    assert np.allclose(grad, [12.0, 4.0], atol=1e-4)

def test_symbolic_gradient_and_hessian():
    x, y = sp.symbols('x y')
    expr_str = "x**2 * y"
    
    # Gradient
    expr, grad = SymbolicCalculusEngine.compute_gradient(expr_str, (x, y))
    grad_val = SymbolicCalculusEngine.evaluate_sym(grad, (x, y), [2.0, 3.0])
    assert np.allclose(grad_val.flatten(), [12.0, 4.0])
    
    # Hessian
    _, hess = SymbolicCalculusEngine.compute_hessian(expr_str, (x, y))
    hess_val = SymbolicCalculusEngine.evaluate_sym(hess, (x, y), [2.0, 3.0])
    # H = [[2y, 2x], [2x, 0]] -> [[6, 4], [4, 0]]
    assert np.allclose(hess_val, [[6.0, 4.0], [4.0, 0.0]])

def test_matrix_calculus_identities():
    X = MatrixEngine([[1, 2], [3, 4]])
    A = MatrixEngine([[1, 0], [0, 1]])
    
    results = MatrixCalculusIdentities.evaluate_identities(X, A)
    
    # d(tr(X)) = I
    assert np.allclose(results['d_trX'], [[1, 0], [0, 1]])
    
    # d(det(X)) = det(X) * X^-T
    # det = -2. X^-1 = [[-2, 1], [1.5, -0.5]]. X^-T = [[-2, 1.5], [1, -0.5]]
    # det * X^-T = [[4, -3], [-2, 1]]
    assert np.allclose(results['d_detX'], [[4, -3], [-2, 1]])

def test_optimization_newton():
    def f_grad(x_arr): return np.array([2*x_arr[0] - 2, 2*x_arr[1] - 4])
    def f_hess(x_arr): return np.array([[2, 0], [0, 2]])
    
    # Starting at (0, 0)
    # Newton should solve quadratic in exactly 1 step (2 states in history: initial, final)
    x_opt, history, conv = OptimizationEngine.newtons_method(f_grad, f_hess, [0.0, 0.0])
    
    assert conv is True
    assert np.allclose(x_opt, [1.0, 2.0])
    assert len(history) == 2
