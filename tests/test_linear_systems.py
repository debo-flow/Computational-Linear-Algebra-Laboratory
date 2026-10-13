import pytest
import numpy as np
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.systems.system_model import LinearSystem
from linear_algebra.systems.classification import SystemClassifier
from linear_algebra.systems.solvers import SystemSolvers
from linear_algebra.systems.verification import SystemVerification

def test_system_initialization_mismatch():
    A = MatrixEngine([[1, 2], [3, 4]])
    B = MatrixEngine([[1], [2], [3]])
    with pytest.raises(ValueError, match="Dimension mismatch"):
        LinearSystem(A, B)

def test_unique_solution_classification():
    A = MatrixEngine([[1, 1], [1, -1]])
    B = MatrixEngine([[3], [1]])
    sys = LinearSystem(A, B)
    report = SystemClassifier.classify(sys)
    assert report["solution_type"] == "unique"

def test_infinite_solution_classification():
    A = MatrixEngine([[1, 1], [2, 2]])
    B = MatrixEngine([[3], [6]])
    sys = LinearSystem(A, B)
    report = SystemClassifier.classify(sys)
    assert report["solution_type"] == "infinite"
    assert "1 free variable" in report["status"]

def test_inconsistent_classification():
    A = MatrixEngine([[1, 1], [2, 2]])
    B = MatrixEngine([[3], [7]])
    sys = LinearSystem(A, B)
    report = SystemClassifier.classify(sys)
    assert report["solution_type"] == "none"

def test_direct_solve_and_verify():
    A = MatrixEngine([[2, 1], [1, 3]])
    B = MatrixEngine([[5], [5]])
    sys = LinearSystem(A, B)
    X = SystemSolvers.direct_solve(sys)
    
    # Verify mathematically
    R, _, total_norm = SystemVerification.compute_residual(sys, X)
    assert total_norm < 1e-9
    
def test_least_squares_overdetermined():
    A = MatrixEngine([[1, 1], [1, -1], [1, 2]])
    B = MatrixEngine([[2], [0], [3]])
    sys = LinearSystem(A, B)
    X, _, _, _ = SystemSolvers.least_squares(sys)
    assert X.shape == (2, 1)
