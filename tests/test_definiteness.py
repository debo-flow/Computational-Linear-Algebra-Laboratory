import pytest
import numpy as np
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.vectors.vector_engine import VectorEngine
from linear_algebra.definiteness.classifier import DefinitenessClassifier
from linear_algebra.definiteness.sylvester import SylvesterCriterion
from linear_algebra.definiteness.quadratic_forms import QuadraticForms
from linear_algebra.definiteness.loewner import LoewnerOrder
from linear_algebra.definiteness.generator import DefiniteMatrixGenerator

def test_pd_eigenvalue_classification():
    # Symmetric Positive Definite
    A = MatrixEngine([[2, -1], [-1, 2]])
    report = DefinitenessClassifier.classify_via_eigenvalues(A)
    assert report['status'] == "Positive Definite (PD)"
    assert report['pos_count'] == 2

def test_psd_classification_with_zero():
    # Positive Semidefinite (Rank 1 matrix)
    A = MatrixEngine([[1, 1], [1, 1]])
    report = DefinitenessClassifier.classify_via_eigenvalues(A)
    assert report['status'] == "Positive Semidefinite (PSD)"
    assert report['zero_count'] == 1

def test_sylvester_criterion():
    A = MatrixEngine([[2, -1], [-1, 2]])
    sylv = SylvesterCriterion.analyze(A)
    assert sylv['is_pd'] is True
    assert sylv['is_nd'] is False
    # Det([2]) = 2, Det(A) = 3
    assert np.allclose(sylv['leading_principal_minors'], [2.0, 3.0])

def test_negative_definite_sylvester():
    A = MatrixEngine([[-2, 1], [1, -2]]) # Evals: -1, -3
    sylv = SylvesterCriterion.analyze(A)
    assert sylv['is_pd'] is False
    assert sylv['is_nd'] is True
    # LPMs should alternate: -2 (k=1), 3 (k=2)
    assert np.allclose(sylv['leading_principal_minors'], [-2.0, 3.0])

def test_quadratic_form_rayleigh():
    A = MatrixEngine([[3, 0], [0, 1]])
    x = VectorEngine([1, 1])
    
    # x^T A x = [1 1] [3, 1]^T = 4
    q_val = QuadraticForms.evaluate(A, x)
    assert np.isclose(q_val, 4.0)
    
    # x^T x = 2
    r_val = QuadraticForms.rayleigh_quotient(A, x)
    assert np.isclose(r_val, 2.0)
    
    # Gradient = 2Ax = 2 * [3, 1]^T = [6, 2]^T
    grad = QuadraticForms.gradient_real(A, x)
    assert np.allclose(grad.vector, [6, 2])

def test_loewner_partial_order():
    A = MatrixEngine([[5, 0], [0, 5]])
    B = MatrixEngine([[2, 1], [1, 2]])
    
    # A - B = [[3, -1], [-1, 3]] -> PD
    comp = LoewnerOrder.compare(A, B)
    assert comp['A_gt_B'] is True
    assert comp['A_geq_B'] is True

def test_generator_consistency():
    # Verify the generator creates accurately bounded matrices
    A_pd = DefiniteMatrixGenerator.generate(3, "PD", seed=42)
    report = DefinitenessClassifier.classify_via_eigenvalues(A_pd)
    assert report['status'] == "Positive Definite (PD)"
