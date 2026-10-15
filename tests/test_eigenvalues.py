import pytest
import numpy as np
import sympy as sp
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.eigenvalues.core import EigenEngine
from linear_algebra.eigenvalues.characteristic import CharacteristicPolynomial
from linear_algebra.eigenvalues.diagnostics import EigenDiagnostics

def test_eigenvalues_identity():
    # Identity matrix has eigenvalue 1 with AM=2, GM=2
    A = MatrixEngine([[1, 0], [0, 1]])
    eigen_data = EigenEngine.compute_eigen_data(A)
    
    assert len(eigen_data) == 1
    assert eigen_data[0]['eigenvalue'] == 1.0
    assert eigen_data[0]['algebraic_multiplicity'] == 2
    assert eigen_data[0]['geometric_multiplicity'] == 2
    assert eigen_data[0]['is_defective'] is False

def test_defective_matrix():
    # Shear matrix: Eval 1, AM=2, GM=1 (defective)
    A = MatrixEngine([[1, 1], [0, 1]])
    eigen_data = EigenEngine.compute_eigen_data(A)
    
    assert len(eigen_data) == 1
    assert eigen_data[0]['eigenvalue'] == 1.0
    assert eigen_data[0]['algebraic_multiplicity'] == 2
    assert eigen_data[0]['geometric_multiplicity'] == 1
    assert eigen_data[0]['is_defective'] is True
    
    is_diag, _ = EigenDiagnostics.check_diagonalizability(A, eigen_data)
    assert is_diag is False

def test_characteristic_polynomial():
    # A = [[1, 2], [3, 4]] -> det(lambda I - A) = lambda^2 - 5*lambda - 2
    A = MatrixEngine([[1, 2], [3, 4]])
    coeffs, roots, is_sym = CharacteristicPolynomial.compute(A)
    
    assert is_sym is False
    # highest power first: [1, -5, -2]
    assert np.allclose(coeffs, [1, -5, -2])

def test_trace_det_identities():
    # Symmetric matrix with evals 3 and -1
    A = MatrixEngine([[1, 2], [2, 1]])
    eigen_data = EigenEngine.compute_eigen_data(A)
    
    sum_valid, prod_valid, s_eig, p_eig, tr, det = EigenDiagnostics.verify_trace_det(A, eigen_data)
    
    assert sum_valid is True
    assert prod_valid is True
    assert np.isclose(tr, 2.0)
    assert np.isclose(det, -3.0)

def test_eigenpair_residuals():
    A = MatrixEngine([[2, 0], [0, 3]])
    eigen_data = EigenEngine.compute_eigen_data(A)
    residuals = EigenDiagnostics.compute_residuals(A, eigen_data)
    
    for res in residuals:
        assert res < 1e-9

def test_nonsquare_matrix_error():
    A = MatrixEngine([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError, match="mathematically defined only for square matrices"):
        EigenEngine.compute_eigen_data(A)
