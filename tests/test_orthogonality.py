import pytest
import numpy as np
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.vectors.vector_engine import VectorEngine
from linear_algebra.orthogonality.inner_product import InnerProductEngine
from linear_algebra.orthogonality.gram_schmidt import GramSchmidtEngine
from linear_algebra.orthogonality.projections import ProjectionEngine
from linear_algebra.svd.pseudoinverse import PseudoinverseEngine

def test_hermitian_inner_product():
    # u = [1, i], v = [1, i]
    # <u, v> = u^H v = (1)(1) + (-i)(i) = 1 - (-1) = 2
    u = VectorEngine([1, 1j])
    v = VectorEngine([1, 1j])
    val = InnerProductEngine.dot(u, v)
    assert np.isclose(val, 2.0 + 0j)

def test_gram_schmidt_linear_dependence():
    # 3 vectors, v3 is a multiple of v1 (dependent)
    v1 = VectorEngine([1, 0, 0])
    v2 = VectorEngine([0, 1, 0])
    v3 = VectorEngine([2, 0, 0])
    
    report = GramSchmidtEngine.compute([v1, v2, v3], method='modified')
    assert len(report['orthonormal_vectors']) == 2
    assert report['skipped_indices'] == [2] # The 3rd vector (index 2) should be skipped
    assert report['orthogonality_error'] < 1e-9

def test_projection_matrix_properties():
    # A random full rank matrix
    A = MatrixEngine([[1, 2], [3, 4], [5, 6]])
    P = ProjectionEngine.projection_matrix(A)
    
    # 1. P^2 = P (Idempotent)
    P_sq = P.matrix @ P.matrix
    assert np.allclose(P_sq, P.matrix)
    
    # 2. P^H = P (Self-Adjoint)
    P_H = P.matrix.conj().T
    assert np.allclose(P_H, P.matrix)

def test_orthogonal_decomposition_pythagorean():
    A = MatrixEngine([[1, 0], [0, 1], [0, 0]]) # Span is XY plane
    v = VectorEngine([3, 4, 5]) # Vector in R3
    
    decomp = ProjectionEngine.orthogonal_decomposition(v, A)
    
    assert np.allclose(decomp['p'].vector, [3, 4, 0])
    assert np.allclose(decomp['r'].vector, [0, 0, 5])
    assert decomp['pythagorean_valid'] is True

def test_least_squares_geometry():
    A = MatrixEngine([[1, 1], [1, -1], [1, 2]])
    b = VectorEngine([2, 0, 3])
    
    A_plus = PseudoinverseEngine.compute(A)
    x_num = np.array(A_plus.matrix) @ np.array(b.vector)
    x = VectorEngine(x_num)
    
    res = ProjectionEngine.verify_least_squares_orthogonality(A, b, x)
    assert res['is_orthogonal'] is True
