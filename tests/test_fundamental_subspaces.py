import pytest
import numpy as np
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.vectors.vector_engine import VectorEngine
from linear_algebra.subspaces.fundamental import FundamentalSubspaces
from linear_algebra.subspaces.orthogonality import SubspaceOrthogonality
from linear_algebra.subspaces.comparison import SubspaceComparison

def test_column_space_uses_original_columns():
    A = MatrixEngine([[1, 2, 3], [2, 4, 6], [1, 0, 1]])
    col_basis, pivots = FundamentalSubspaces.get_column_space(A)
    # The first and second columns are independent
    assert pivots == [0, 1]
    assert len(col_basis) == 2
    # Ensure it pulled from original matrix
    assert col_basis[0].vector.tolist() == [1.0, 2.0, 1.0]

def test_rank_nullity_domain_and_codomain():
    # 2x3 matrix: rank 2, nullity 1, left nullity 0
    A = MatrixEngine([[1, 2, 3], [0, 1, 4]])
    report = FundamentalSubspaces.analyze(A)
    assert report["rank"] == 2
    assert report["nullity"] == 1
    assert report["left_nullity"] == 0
    assert report["rank_nullity_domain"] is True
    assert report["rank_nullity_codomain"] is True

def test_null_space_residuals():
    A = MatrixEngine([[1, -1], [2, -2]])
    report = FundamentalSubspaces.analyze(A)
    # Null space should have dimension 1
    assert report["null_space"]["dim"] == 1
    # Check that maximum residual is near zero
    assert report["null_space"]["max_residual"] < 1e-9

def test_orthogonality():
    A = MatrixEngine([[1, 1], [1, 1]])
    report = FundamentalSubspaces.analyze(A)
    # Row Space = Span([1,1])
    # Null Space = Span([-1,1])
    is_ortho, res = SubspaceOrthogonality.verify_orthogonality(
        report["row_space"]["basis"], 
        report["null_space"]["basis"]
    )
    assert is_ortho is True
    assert res < 1e-9

def test_subspace_comparison():
    U1 = VectorEngine([1, 0, 0])
    U2 = VectorEngine([0, 1, 0])
    
    V1 = VectorEngine([1, 1, 0])
    V2 = VectorEngine([1, -1, 0])
    
    # Both sets span the XY plane
    report = SubspaceComparison.compare_bases([U1, U2], [V1, V2])
    assert report["spans_same"] is True
    assert report["dim_U"] == 2
    assert report["dim_V"] == 2
    assert report["dim_sum_UV"] == 2
    
def test_zero_matrix_subspaces():
    A = MatrixEngine([[0, 0], [0, 0]])
    report = FundamentalSubspaces.analyze(A)
    assert report["rank"] == 0
    assert report["nullity"] == 2
    assert report["left_nullity"] == 2
