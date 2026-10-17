import pytest
import numpy as np
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.svd.core import SVDEngine
from linear_algebra.svd.approximation import LowRankApproximation
from linear_algebra.svd.pseudoinverse import PseudoinverseEngine

def test_full_svd_reconstruction():
    A = MatrixEngine([[1, 2], [3, 4], [5, 6]])
    data = SVDEngine.compute(A, full_matrices=True)
    A_recon = SVDEngine.reconstruct(data['U'], data['S'], data['Vh'])
    
    diff = np.linalg.norm(np.array(A.matrix, dtype=float) - np.array(A_recon.matrix, dtype=float))
    assert diff < 1e-9

def test_rank_deficient_condition():
    # Rank 1 matrix, condition number should be infinite mathematically, 
    # but we trap it and set inf or very high
    A = MatrixEngine([[1, 1], [1, 1]])
    data = SVDEngine.compute(A)
    assert data['rank'] == 1
    assert data['condition_number'] == float('inf')

def test_eckart_young_low_rank():
    A = MatrixEngine([[1, 0, 0], [0, 2, 0], [0, 0, 3]]) # SVD diag: 3, 2, 1
    # Rank 2 approximation strips the '1'
    approx = LowRankApproximation.truncate(A, 2)
    assert approx['rank_k'] == 2
    # The theoretical frob error is sqrt(1^2) = 1.0
    assert np.isclose(approx['theoretical_frob_error'], 1.0)
    assert np.isclose(approx['abs_error_frob'], 1.0)

def test_moore_penrose_conditions():
    # Rectangular matrix
    A = MatrixEngine([[1, 2], [3, 4], [5, 6]])
    A_plus = PseudoinverseEngine.compute(A)
    checks = PseudoinverseEngine.verify_mp_conditions(A, A_plus)
    assert checks['all_valid'] is True

def test_svd_least_squares_minimum_norm():
    # Underdetermined system: x + y = 1 -> infinite solutions
    # The minimum norm solution is x=0.5, y=0.5 (norm = 0.707)
    A = MatrixEngine([[1, 1]])
    b = MatrixEngine([[1]])
    x, res_norm, sol_norm = PseudoinverseEngine.svd_least_squares(A, b)
    
    assert np.isclose(res_norm, 0.0) # exact solution exists
    assert np.allclose(x.matrix, [[0.5], [0.5]])
    assert np.isclose(sol_norm, np.sqrt(0.5**2 + 0.5**2))
