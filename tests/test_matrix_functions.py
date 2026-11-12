import pytest
import numpy as np
import scipy.linalg as spla
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.vectors.vector_engine import VectorEngine
from linear_algebra.matrix_functions.core import MatrixFunctionsEngine
from linear_algebra.matrix_functions.diagnostics import FunctionDiagnostics
from linear_algebra.matrix_functions.ode_solver import LinearODESolver

def test_matrix_exponential_diagonal():
    A = MatrixEngine([[2, 0], [0, -3]])
    exp_A = MatrixFunctionsEngine.expm(A)
    
    expected = np.array([[np.exp(2), 0], [0, np.exp(-3)]])
    assert np.allclose(exp_A.matrix, expected)

def test_matrix_exponential_nilpotent():
    # Nilpotent matrix: N^2 = 0. So e^N = I + N
    N = MatrixEngine([[0, 1], [0, 0]])
    exp_N = MatrixFunctionsEngine.expm(N)
    
    expected = np.array([[1, 1], [0, 1]])
    assert np.allclose(exp_N.matrix, expected)

def test_matrix_logarithm_reconstruction():
    A = MatrixEngine([[2, 1], [0, 3]])
    rel_err, recon_A = FunctionDiagnostics.verify_exp_log(A)
    assert rel_err < 1e-9

def test_matrix_singular_logarithm_error():
    # Singular matrix, log diverges
    A = MatrixEngine([[1, 1], [1, 1]])
    with pytest.raises(ValueError, match="singular"):
        MatrixFunctionsEngine.logm(A)

def test_matrix_square_root_reconstruction():
    A = MatrixEngine([[4, 0], [0, 9]])
    rel_err, recon_A = FunctionDiagnostics.verify_sqrtm(A)
    assert rel_err < 1e-9

def test_fractional_power():
    A = MatrixEngine([[4, 0], [0, 9]])
    A_half = MatrixFunctionsEngine.fractional_power(A, 0.5)
    # A^0.5 should equal sqrt(A)
    assert np.allclose(A_half.matrix, [[2, 0], [0, 3]])

def test_general_functions():
    A = MatrixEngine([[0, 0], [0, 0]])
    res_cos = MatrixFunctionsEngine.general_function(A, 'cos')
    # cos(0) = I
    assert np.allclose(res_cos.matrix, [[1, 0], [0, 1]])
    
    res_sin = MatrixFunctionsEngine.general_function(A, 'sin')
    # sin(0) = 0
    assert np.allclose(res_sin.matrix, [[0, 0], [0, 0]])

def test_eigen_based_function():
    A = MatrixEngine([[1, 0], [0, 2]])
    res, report = MatrixFunctionsEngine.eigen_based_function(A, np.exp)
    
    assert np.allclose(res.matrix, [[np.exp(1), 0], [0, np.exp(2)]])

def test_eigen_based_function_defective():
    # Defective matrix cannot be diagonalized
    A = MatrixEngine([[1, 1], [0, 1]])
    with pytest.raises(ValueError, match="defective"):
        MatrixFunctionsEngine.eigen_based_function(A, np.exp)

def test_linear_ode_solver():
    A = MatrixEngine([[0, 1], [-1, 0]]) # Harmonic oscillator
    x0 = VectorEngine([1, 0])
    times, trajs = LinearODESolver.solve_trajectory(A, x0, t_max=np.pi, steps=10)
    
    # At t = pi, x(t) should be approx [-1, 0] because e^{pi * A} = [[-1, 0], [0, -1]]
    final_state = trajs[-1]
    assert np.allclose(final_state, [-1, 0], atol=1e-4)
