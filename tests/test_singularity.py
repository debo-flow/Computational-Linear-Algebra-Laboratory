import pytest
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.matrix.singularity_analysis import SingularityAnalyzer

def test_exact_singular_matrix():
    A = MatrixEngine([[1, 2], [2, 4]])
    report = SingularityAnalyzer.analyze(A)
    assert report["is_singular"] is True
    assert report["rank"] == 1
    assert "Singular" in report["status"]

def test_well_conditioned_matrix():
    A = MatrixEngine([[1, 0], [0, 1]])
    report = SingularityAnalyzer.analyze(A)
    assert report["is_singular"] is False
    assert report["rank"] == 2
    assert "Well-Conditioned" in report["status"]

def test_rectangular_matrix_singularity():
    A = MatrixEngine([[1, 2, 3], [4, 5, 6]])
    report = SingularityAnalyzer.analyze(A)
    assert "Not Applicable" in report["status"]
