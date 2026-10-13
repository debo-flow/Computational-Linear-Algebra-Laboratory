import pytest
import numpy as np
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.matrix.advanced_operations import AdvancedMatrixOperations

def test_hadamard_product():
    A = MatrixEngine([[1, 2], [3, 4]])
    B = MatrixEngine([[2, 0], [1, 2]])
    C = AdvancedMatrixOperations.hadamard_product(A, B)
    assert C.matrix.tolist() == [[2, 0], [3, 8]]

def test_elementwise_division_zero():
    A = MatrixEngine([[1, 2]])
    B = MatrixEngine([[1, 0]])
    with pytest.raises(ValueError, match="Division by near-zero"):
        AdvancedMatrixOperations.elementwise_division(A, B)

def test_matrix_power():
    A = MatrixEngine([[1, 2], [0, 1]])
    A3 = AdvancedMatrixOperations.matrix_power(A, 3)
    assert A3.matrix.tolist() == [[1, 6], [0, 1]]
    
def test_kronecker_product_dimensions():
    A = MatrixEngine([[1, 2]])          # 1x2
    B = MatrixEngine([[1, 2], [3, 4]])  # 2x2
    C = AdvancedMatrixOperations.kronecker_product(A, B)
    assert C.shape == (2, 4)
