import pytest
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.matrix.analysis import MatrixAnalysis

def test_norms():
    A = MatrixEngine([[-3, 4], [0, 0]])
    norms = MatrixAnalysis.norms(A)
    assert norms["Frobenius Norm"] == 5.0
    assert norms["Infinity-Norm (Max Row Sum)"] == 7.0

def test_nilpotent_structure():
    A = MatrixEngine([[0, 1], [0, 0]])
    structs = MatrixAnalysis.structure_tests(A)
    assert structs["Nilpotent"] is True
    assert structs["Idempotent"] is False

def test_linear_independence():
    A = MatrixEngine([[1, 2], [2, 4]])  # Dependent cols
    analysis = MatrixAnalysis.linear_independence(A)
    assert analysis["Rank"] == 1
    assert analysis["Full Column Rank (Independent Cols)"] is False
