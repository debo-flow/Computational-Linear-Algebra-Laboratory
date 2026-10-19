import pytest
import numpy as np
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.qr.core import QREngine
from linear_algebra.qr.least_squares import QRLeastSquares
from linear_algebra.qr.educational import QREducational

def test_library_full_qr():
    # 3x2 Matrix (Rectangular)
    A = MatrixEngine([[1, 2], [3, 4], [5, 6]])
    data = QREngine.compute_library_qr(A, mode='full')
    
    # Q should be 3x3, R should be 3x2
    assert data['Q'].shape == (3, 3)
    assert data['R'].shape == (3, 2)
    assert data['reconstruction_error'] < 1e-9
    assert data['orthogonality_error'] < 1e-9

def test_library_reduced_qr():
    A = MatrixEngine([[1, 2], [3, 4], [5, 6]])
    data = QREngine.compute_library_qr(A, mode='reduced')
    
    # Q should be 3x2, R should be 2x2
    assert data['Q'].shape == (3, 2)
    assert data['R'].shape == (2, 2)
    assert data['reconstruction_error'] < 1e-9

def test_pivoted_qr_rank_revealing():
    # Rank 1 matrix (Columns are multiples)
    A = MatrixEngine([[1, 2, 3], [2, 4, 6], [1, 2, 3]])
    data = QREngine.compute_pivoted_qr(A)
    
    assert data['rank'] == 1
    # First diagonal element is large, subsequent are ~0
    assert data['diag_R'][0] > 1.0
    assert data['diag_R'][1] < 1e-9

def test_qr_least_squares_multiple_rhs():
    # A is 3x2
    A = MatrixEngine([[1, 1], [1, -1], [1, 2]])
    # B is 3x2 (Two RHS)
    B = MatrixEngine([[2, 4], [0, 0], [3, 6]])
    
    res = QRLeastSquares.solve(A, B)
    assert res['status'] == "success"
    assert res['X'].shape == (2, 2)
    assert len(res['residual_norms']) == 2
    assert res['overall_residual'] > 0 # Overdetermined inconsistent

def test_qr_least_squares_rank_deficient_fallback():
    # A is 2x2 but rank 1
    A = MatrixEngine([[1, 1], [1, 1]])
    B = MatrixEngine([[2], [3]])
    res = QRLeastSquares.solve(A, B)
    assert res['status'] == 'rank_deficient'

def test_weighted_least_squares():
    A = MatrixEngine([[1, 1], [1, -1], [1, 2]])
    B = MatrixEngine([[2], [0], [3]])
    # Put massive weight on the second equation (A[1]*x = B[1] -> x - y = 0)
    res = QRLeastSquares.solve_weighted(A, B, weights=[1, 1e6, 1])
    
    # The solver must heavily prioritize x - y = 0
    x_val = res['X'].matrix[0, 0]
    y_val = res['X'].matrix[1, 0]
    assert np.isclose(x_val, y_val, atol=1e-3)

def test_householder_educational():
    A = MatrixEngine([[1, 2], [3, 4], [5, 6]])
    Q, R, steps = QREngine.householder_qr(A)
    
    assert len(steps) == 2 # 2 columns to eliminate
    # Verify A = QR
    A_recon = Q.matrix @ R.matrix
    assert np.linalg.norm(A_recon - A.matrix) < 1e-9
