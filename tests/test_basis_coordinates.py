import pytest
import numpy as np
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.vectors.vector_engine import VectorEngine
from linear_algebra.vector_spaces.basis import BasisAnalyzer
from linear_algebra.vector_spaces.dimension import DimensionAnalyzer
from linear_algebra.vector_spaces.coordinates import CoordinateSystem

def test_extract_basis():
    # 3 vectors in R^2, 2 are independent
    v1 = VectorEngine([1, 0])
    v2 = VectorEngine([0, 1])
    v3 = VectorEngine([2, 3]) # Redundant
    
    basis, pivots = BasisAnalyzer.extract_basis([v1, v2, v3])
    assert len(basis) == 2
    assert pivots == [0, 1]

def test_ambient_basis_validation():
    v1 = VectorEngine([1, 0])
    v2 = VectorEngine([0, 1])
    is_valid, _ = BasisAnalyzer.validate_ambient_basis([v1, v2], target_dim=2)
    assert is_valid is True

    # Too few vectors
    is_valid, _ = BasisAnalyzer.validate_ambient_basis([v1], target_dim=2)
    assert is_valid is False

def test_rank_nullity():
    # 2x3 Matrix. Rank = 2, Nullity = 1
    A = MatrixEngine([[1, 2, 3], [4, 5, 6]])
    report = DimensionAnalyzer.analyze_matrix_spaces(A)
    assert report["rank"] == 2
    assert report["nullity"] == 1
    assert report["rank_nullity_holds"] is True

def test_get_coordinates():
    # Basis: [1,1] and [1,-1]
    b1 = VectorEngine([1, 1])
    b2 = VectorEngine([1, -1])
    v = VectorEngine([3, 1])
    
    coords, res = CoordinateSystem.get_coordinates([b1, b2], v)
    assert res < 1e-9
    assert coords.matrix.tolist() == [[2.0], [1.0]] # 2(1,1) + 1(1,-1) = (3,1)

def test_change_of_basis():
    # Standard basis to standard basis should be Identity
    B = [VectorEngine([1, 0]), VectorEngine([0, 1])]
    C = [VectorEngine([1, 0]), VectorEngine([0, 1])]
    
    P_CB = CoordinateSystem.change_of_basis_matrix(B, C)
    assert np.allclose(P_CB.matrix, np.eye(2))
    
def test_target_outside_span():
    b1 = VectorEngine([1, 0, 0])
    b2 = VectorEngine([0, 1, 0])
    v = VectorEngine([0, 0, 5]) # Outside XY plane
    with pytest.raises(ValueError, match="OUTSIDE the span"):
        CoordinateSystem.get_coordinates([b1, b2], v)
