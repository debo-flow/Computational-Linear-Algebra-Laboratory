import pytest
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.matrix.determinant_properties import DeterminantProperties

def test_transpose_property():
    A = MatrixEngine([[3, 1], [2, 4]])
    is_valid, _, _, _ = DeterminantProperties.verify_transpose(A)
    assert is_valid is True

def test_multiplication_property():
    A = MatrixEngine([[3, 1], [2, 4]])
    B = MatrixEngine([[1, 5], [0, 2]])
    is_valid, _, _, _ = DeterminantProperties.verify_multiplication(A, B)
    assert is_valid is True
