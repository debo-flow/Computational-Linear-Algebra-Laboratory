import pytest
import numpy as np
import scipy.sparse as sps
from linear_algebra.sparse.core import SparseMatrixEngine
from linear_algebra.sparse.solvers import SparseSolverEngine
from linear_algebra.sparse.eigen import SparseEigenEngine
from linear_algebra.sparse.applications import SparseApplications

def test_sparse_creation_and_memory():
    A = SparseMatrixEngine.create_random(100, 100, density=0.05, format='csr', seed=42)
    assert sps.isspmatrix_csr(A)
    assert A.shape == (100, 100)
    
    report = SparseMatrixEngine.memory_analysis(A)
    assert report['shape'] == (100, 100)
    assert report['compression_ratio'] > 1.0 # Must be heavily compressed

def test_sparse_format_conversion():
    A = SparseMatrixEngine.create_random(10, 10, density=0.2, format='coo', seed=1)
    A_csc = SparseMatrixEngine.convert_format(A, 'csc')
    
    assert sps.isspmatrix_csc(A_csc)
    # Ensure data integrity survived conversion
    assert np.allclose(A.toarray(), A_csc.toarray())

def test_sparse_direct_solve():
    A = sps.eye(50, format='csr') * 2.0
    b = np.ones(50)
    
    x, res, time_taken = SparseSolverEngine.direct_solve(A, b)
    
    # 2x = 1 -> x = 0.5
    assert np.allclose(x, 0.5)
    assert res < 1e-9

def test_sparse_iterative_cg():
    # SPD Matrix
    A = sps.diags([np.ones(49)*-1, np.ones(50)*2, np.ones(49)*-1], [-1, 0, 1], format='csr')
    b = np.ones(50)
    
    res = SparseSolverEngine.iterative_solve(A, b, method='cg')
    assert "Converged" in res['status']
    assert res['residual_norm'] < 1e-5

def test_sparse_eigen_arpack():
    # Tridiagonal matrix eigenvalues are analytically known, but we test solver integrity
    A = sps.diags([np.ones(99), np.ones(100)*2, np.ones(99)], [-1, 0, 1], format='csr')
    
    # Ask for 3 largest magnitude
    res = SparseEigenEngine.compute_eigs(A, k=3, which='LM')
    assert "Converged" in res['status']
    assert len(res['eigenvalues']) == 3
    
    for r in res['residuals']:
        assert r < 1e-4

def test_application_laplacian_2d():
    L = SparseApplications.laplacian_2d(5, 5)
    assert L.shape == (25, 25)
    assert sps.isspmatrix_csr(L)
