import pytest
import numpy as np
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.matrix.decompositions import MatrixDecompositions
from linear_algebra.matrix.validation import ValidationEngine

def test_lu_reconstruction():
    A = MatrixEngine([[4, 3], [6, 3]])
    P, L, U = MatrixDecompositions.lu_decomposition(A)
    is_valid, residual, _ = ValidationEngine.verify_lu(A, P, L, U)
    assert is_valid is True
    assert residual < 1e-9

def test_cholesky_not_positive_definite():
    # Symmetric but not positive definite (eigenvalues contain negatives)
    A = MatrixEngine([[1, 2], [2, 1]])
    with pytest.raises(ValueError, match="positive-definite"):
        MatrixDecompositions.cholesky_decomposition(A)
