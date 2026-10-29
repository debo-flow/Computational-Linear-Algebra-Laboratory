import pytest
import numpy as np
import scipy.linalg as spla
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.vectors.vector_engine import VectorEngine
from linear_algebra.numerical_eigen.iterative import IterativeEigenEngine
from linear_algebra.numerical_eigen.qr_algorithm import QREigenAlgorithm
from linear_algebra.numerical_eigen.schur import SchurDecomposition
from linear_algebra.numerical_eigen.sensitivity import EigenSensitivity

def test_power_iteration_dominant():
    # Matrix with eigenvalues 3, 2. Dominant is 3.
    A = MatrixEngine([[3, 0], [0, 2]])
    x0 = VectorEngine([1, 1])
    lam, vec, hist, conv = IterativeEigenEngine.power_iteration(A, x0)
    assert conv is True
    assert np.isclose(lam, 3.0)
    assert np.isclose(abs(vec.vector[0]), 1.0) # Eigenvector should be [1, 0]
    assert np.isclose(vec.vector[1], 0.0, atol=1e-7)

def test_inverse_iteration_shift():
    # Eigenvalues 5, 1. Shift near 1 should find 1.
    A = MatrixEngine([[5, 0], [0, 1]])
    x0 = VectorEngine([1, 1])
    lam, vec, hist, conv = IterativeEigenEngine.inverse_iteration(A, x0, shift=1.1)
    assert conv is True
    assert np.isclose(lam, 1.0)

def test_rqi_convergence():
    A = MatrixEngine([[4, 1], [1, 3]])
    x0 = VectorEngine([1, 0])
    lam, vec, hist, conv = IterativeEigenEngine.rayleigh_quotient_iteration(A, x0)
    assert conv is True
    # Verify residual is near 0
    res = np.linalg.norm(np.array(A.matrix) @ np.array(vec.vector) - lam * np.array(vec.vector))
    assert res < 1e-9

def test_schur_decomposition_complex():
    A = MatrixEngine([[1, -1], [1, 1]]) # Eigenvalues 1+i, 1-i
    res = SchurDecomposition.compute(A, output='complex')
    
    # T is strictly upper triangular in complex Schur form
    T = np.array(res['T'].matrix)
    assert np.isclose(T[1, 0], 0.0)
    
    # Eigenvalues on diagonal
    evals = np.diag(T)
    assert np.isclose(evals[0], 1 + 1j) or np.isclose(evals[0], 1 - 1j)
    
    # Valid reconstruction
    assert res['reconstruction_error'] < 1e-9
    assert res['unitarity_error'] < 1e-9

def test_eigenvalue_sensitivity():
    # Non-normal matrix, highly sensitive
    A = MatrixEngine([[1, 1000], [0, 1]])
    conds = EigenSensitivity.compute_condition_numbers(A)
    # Condition number should be very high (eigenvectors are highly non-orthogonal)
    assert conds[0]['condition_number'] > 1e2

def test_qr_educational_convergence():
    # Symmetric matrix converges nicely
    A = MatrixEngine([[2, 1], [1, 2]])
    A_inf, hist = QREigenAlgorithm.educational_qr_iteration(A, max_iter=50)
    
    T = np.array(A_inf.matrix)
    assert np.isclose(T[1, 0], 0.0, atol=1e-4) # off diagonal converges to 0
    evals = np.sort(np.diag(T))
    assert np.allclose(evals, [1.0, 3.0], atol=1e-4)
