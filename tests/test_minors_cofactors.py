import pytest
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.matrix.minors_cofactors import MinorsCofactors
from linear_algebra.matrix.operations import MatrixOperations

def test_minor_matrix():
    A = MatrixEngine([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    M_11 = MinorsCofactors.minor_matrix(A, 1, 1) # Delete row 1, col 1 (0-indexed)
    assert M_11.matrix.tolist() == [[1, 3], [7, 9]]

def test_adjugate_identity():
    A = MatrixEngine([[1, 2], [3, 4]])
    adj = MinorsCofactors.adjugate(A)
    # Validate A * adj(A) = det(A) * I
    A_adj = MatrixOperations.multiply(A, adj)
    det = MatrixOperations.determinant(A)
    assert A_adj.matrix.tolist() == [[det, 0], [0, det]]

def test_1x1_cofactor():
    A = MatrixEngine([[5]])
    C = MinorsCofactors.cofactor_matrix(A)
    assert C.matrix.tolist() == [[1]]
