import pytest
import numpy as np
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.vectors.vector_engine import VectorEngine
from linear_algebra.generalized_eigen.core import GeneralizedEigenEngine
from linear_algebra.generalized_eigen.pencil import MatrixPencilAnalyzer
from linear_algebra.generalized_eigen.rayleigh import GeneralizedRayleighQuotient

def test_general_gep_finite():
    # A = [[4,0], [0,4]], B = [[2,0], [0,2]] -> lambda = 2, 2
    A = MatrixEngine([[4, 0], [0, 4]])
    B = MatrixEngine([[2, 0], [0, 2]])
    res = GeneralizedEigenEngine.compute_general(A, B)
    
    assert len(res) == 2
    assert res[0]['status'] == "Finite"
    assert np.isclose(res[0]['eigenvalue'], 2.0)
    assert res[0]['residual'] < 1e-9

def test_general_gep_infinite():
    # A = [[1,0], [0,1]], B = [[1,0], [0,0]] -> lambda = 1, inf
    # Node 2 is "massless"
    A = MatrixEngine([[1, 0], [0, 1]])
    B = MatrixEngine([[1, 0], [0, 0]])
    res = GeneralizedEigenEngine.compute_general(A, B)
    
    statuses = [r['status'] for r in res]
    assert "Finite" in statuses
    assert "Infinite (Singular B)" in statuses
    
    for r in res:
        if "Infinite" in r['status']:
            assert np.isclose(r['beta'], 0.0, atol=1e-9)
            assert r['residual'] < 1e-9

def test_symmetric_definite_solver():
    A = MatrixEngine([[2, -1], [-1, 2]]) # PD Stiffness
    B = MatrixEngine([[3, 0], [0, 3]])   # PD Mass
    
    res, b_ortho = GeneralizedEigenEngine.compute_symmetric_definite(A, B)
    
    assert len(res) == 2
    # Check B-Orthogonality
    assert b_ortho < 1e-9
    
    for r in res:
        assert r['residual'] < 1e-9

def test_symmetric_definite_failure_on_singular_B():
    A = MatrixEngine([[2, -1], [-1, 2]])
    B = MatrixEngine([[1, 0], [0, 0]]) # B is not Positive-Definite
    
    with pytest.raises(ValueError, match="Matrix B is not Positive-Definite"):
        GeneralizedEigenEngine.compute_symmetric_definite(A, B)

def test_pencil_diagnostics():
    A = MatrixEngine([[1, 2], [3, 4]])
    B = MatrixEngine([[1, 0], [0, 0]])
    diag = MatrixPencilAnalyzer.analyze_pencil(A, B)
    
    assert diag['is_B_singular'] is True
    assert diag['rank_B'] == 1
    assert diag['cond_B'] == float('inf')

def test_generalized_rayleigh_quotient():
    A = MatrixEngine([[2, 0], [0, 4]])
    B = MatrixEngine([[1, 0], [0, 2]])
    x = VectorEngine([1, 1])
    
    # x^T A x = [1 1] [2, 4]^T = 6
    # x^T B x = [1 1] [1, 2]^T = 3
    # R(x) = 6/3 = 2.0
    val = GeneralizedRayleighQuotient.compute(A, B, x)
    assert np.isclose(val, 2.0)
    
def test_rayleigh_zero_denominator():
    A = MatrixEngine([[1, 0], [0, 1]])
    B = MatrixEngine([[0, 0], [0, 0]])
    x = VectorEngine([1, 1])
    
    with pytest.raises(ValueError, match="denominator"):
        GeneralizedRayleighQuotient.compute(A, B, x)
