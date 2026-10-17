import pytest
import numpy as np
import sympy as sp
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.diagonalization.core import DiagonalizationEngine
from linear_algebra.diagonalization.similarity import SimilarityAnalyzer
from linear_algebra.diagonalization.functions import MatrixFunctions
from linear_algebra.diagonalization.jordan import JordanCanonicalForm

def test_numeric_diagonalization():
    # Symmetric matrix is always diagonalizable
    A = MatrixEngine([[2, 1], [1, 2]])
    report = DiagonalizationEngine.diagonalize(A)
    assert report['is_diagonalizable'] is True
    assert report['residual'] < 1e-9

def test_defective_matrix():
    # Shear matrix cannot be diagonalized
    A = MatrixEngine([[1, 1], [0, 1]], is_symbolic=True)
    report = DiagonalizationEngine.diagonalize(A)
    assert report['is_diagonalizable'] is False
    assert "defective" in report['message'].lower()

def test_similarity_transform():
    A = MatrixEngine([[1, 2], [3, 4]])
    P = MatrixEngine([[0, 1], [1, 0]]) # Swap basis
    B, P_inv = SimilarityAnalyzer.transform(A, P)
    # P^-1 A P should swap rows and columns
    assert np.allclose(B.matrix, [[4, 3], [2, 1]])

def test_matrix_power_diag():
    A = MatrixEngine([[2, 0], [0, 3]])
    report = DiagonalizationEngine.diagonalize(A)
    A_3 = MatrixFunctions.matrix_power_diag(report['P'], report['D'], report['P_inv'], 3)
    # A^3 should be [[8, 0], [0, 27]]
    assert np.allclose(A_3.matrix, [[8, 0], [0, 27]])

def test_jordan_canonical_form():
    # Symbolic defective matrix
    A = MatrixEngine([[5, 4, 2, 1], [0, 1, -1, -1], [-1, -1, 3, 0], [1, 1, -1, 2]], is_symbolic=True)
    P, J = JordanCanonicalForm.compute(A)
    # SymPy output J should have 1s on superdiagonal for defective blocks
    # Ensure P J P^-1 == A
    A_recon = sp.Matrix(P.matrix) * sp.Matrix(J.matrix) * sp.Matrix(P.matrix).inv()
    diff = sp.simplify(A_recon - sp.Matrix(A.matrix))
    assert diff.is_zero_matrix
