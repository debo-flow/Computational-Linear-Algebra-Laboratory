import pytest
import numpy as np
from linear_algebra.optimization.functions import BenchmarkFunctions
from linear_algebra.optimization.unconstrained import UnconstrainedOptimization
from linear_algebra.optimization.constrained import ConstrainedOptimization
from linear_algebra.optimization.least_squares import NonlinearLeastSquares

def test_benchmark_functions_gradient():
    # Sphere function at (1, 2) -> val=5, grad=[2, 4], hess=[[2,0],[0,2]]
    x = np.array([1.0, 2.0])
    val, grad, hess = BenchmarkFunctions.sphere(x)
    assert np.isclose(val, 5.0)
    assert np.allclose(grad, [2.0, 4.0])
    assert np.allclose(hess, [[2.0, 0.0], [0.0, 2.0]])

def test_gradient_descent_convergence():
    x0 = np.array([5.0, -5.0])
    x_opt, hist, conv = UnconstrainedOptimization.gradient_descent(
        BenchmarkFunctions.sphere, x0, lr=0.1, max_iter=200
    )
    assert conv is True
    assert np.allclose(x_opt, [0.0, 0.0], atol=1e-3)

def test_newtons_method_quadratic():
    # Newton's method should solve a quadratic (Sphere) in exactly 1 step
    x0 = np.array([10.0, -10.0])
    x_opt, hist, conv = UnconstrainedOptimization.newtons_method(
        BenchmarkFunctions.sphere, x0, max_iter=10
    )
    assert conv is True
    assert len(hist) <= 3  # Initial point, step 1 (solution), maybe a convergence check step
    assert np.allclose(x_opt, [0.0, 0.0], atol=1e-7)

def test_bfgs_optimization():
    x0 = np.array([-1.2, 1.0])
    x_opt, hist, conv = UnconstrainedOptimization.bfgs(
        BenchmarkFunctions.rosenbrock, x0, max_iter=500
    )
    # Global minimum of Rosenbrock is at (1, 1)
    assert conv is True
    assert np.allclose(x_opt, [1.0, 1.0], atol=1e-4)

def test_quadratic_penalty_method():
    # Minimize x^2 + y^2 subject to x + y - 1 = 0
    # True constrained minimum is at (0.5, 0.5)
    def eq_constraint(x):
        return (x[0] + x[1] - 1), np.array([1.0, 1.0]), np.zeros((2, 2))
        
    x0 = np.array([0.0, 0.0])
    x_opt, hist, conv = ConstrainedOptimization.quadratic_penalty_method(
        BenchmarkFunctions.sphere, [eq_constraint], [], x0, mu=1e4
    )
    
    assert np.allclose(x_opt, [0.5, 0.5], atol=1e-3)

def test_levenberg_marquardt_nlls():
    def res_func(x):
        return np.array([x[0]**2 - 4, x[1]**2 - 9])
    def jac_func(x):
        return np.array([[2*x[0], 0], [0, 2*x[1]]])
        
    x0 = np.array([1.0, 1.0])
    x_opt, hist, conv = NonlinearLeastSquares.levenberg_marquardt(res_func, jac_func, x0)
    
    assert conv is True
    # Roots are at (+-2, +-3), given positive starting point, should converge to (2, 3)
    assert np.allclose(x_opt, [2.0, 3.0], atol=1e-5)
