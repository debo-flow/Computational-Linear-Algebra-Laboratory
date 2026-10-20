import pytest
import numpy as np
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.factorizations.lu import LUEngine
from linear_algebra.factorizations.cholesky import CholeskyEngine
from linear_algebra.factorizations.triangular_solver import TriangularSolver

def test_lu_partial_pivoting():
    # Matrix that requires a row swap to avoid a zero pivot
    A = MatrixEngine([[0, 1], [2, 3]])
    report = LUEngine.compute_partial_pivot(A)
    
    # Verify P^T @ L @ U = A  => P A = L U
    A_recon = report['P'].matrix.T @ report['L'].matrix @ report['U'].matrix
    assert np.allclose(A_recon, A.matrix)
    assert report['reconstruction_error'] < 1e-9

def test_lu_no_pivoting_failure():
    A = MatrixEngine([[0, 1], [2, 3]])
    with pytest.raises(ValueError, match="Zero or near-zero pivot"):
        LUEngine.compute_no_pivot(A)

def test_cholesky_positive_definite():
    # Symmetric Positive-Definite Matrix
    A = MatrixEngine([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])
    report = CholeskyEngine.compute_cholesky(A)
    
    L = report['L'].matrix
    assert np.allclose(L @ L.conj().T, A.matrix)
    assert report['reconstruction_error'] < 1e-9

def test_cholesky_indefinite_failure():
    # Symmetric Indefinite Matrix (eigenvalues mix of positive/negative)
    A = MatrixEngine([[1, 2], [2, 1]])
    with pytest.raises(ValueError, match="not Positive-Definite"):
        CholeskyEngine.compute_cholesky(A)

def test_ldl_indefinite_success():
    # Symmetric Indefinite Matrix (fails Cholesky, succeeds LDL^T)
    A = MatrixEngine([[1, 2], [2, 1]])
    report = CholeskyEngine.compute_ldl(A)
    
    # A = P L D L^H P^T
    A_recon = report['P'].matrix @ report['L'].matrix @ report['D'].matrix @ report['L'].matrix.conj().T @ report['P'].matrix.T
    assert np.allclose(A_recon, A.matrix)
    assert report['reconstruction_error'] < 1e-9

def test_triangular_solver_pipeline():
    A = MatrixEngine([[2, 1], [1, 3]])
    b = MatrixEngine([[5], [5]])
    
    # Factorize
    lu_data = LUEngine.compute_partial_pivot(A)
    P, L, U = lu_data['P'], lu_data['L'], lu_data['U']
    
    # Solve
    x, y, Pb = TriangularSolver.solve_via_lu(P, L, U, b)
    
    # Truth: x = [2, 1]^T
    assert np.allclose(x.matrix, [[2], [1]])
    
    # Verify Residual
    Ax = A.matrix @ x.matrix
    res_norm = np.linalg.norm(b.matrix - Ax)
    assert res_norm < 1e-9
