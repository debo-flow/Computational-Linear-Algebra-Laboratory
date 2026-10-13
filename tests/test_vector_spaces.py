import pytest
import numpy as np
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.vectors.vector_engine import VectorEngine
from linear_algebra.vector_spaces.linear_combinations import LinearCombinations
from linear_algebra.vector_spaces.span_analysis import SpanAnalyzer
from linear_algebra.vector_spaces.independence import LinearIndependenceAnalyzer
from linear_algebra.vector_spaces.subspace_analysis import SubspaceAnalyzer
from linear_algebra.vector_spaces.axiom_checker import AxiomExplorer

def test_linear_combinations():
    v1 = VectorEngine([1, 0])
    v2 = VectorEngine([0, 1])
    res = LinearCombinations.compute([2, -3], [v1, v2])
    assert list(res.vector) == [2.0, -3.0]

def test_linear_combination_mismatch():
    v1 = VectorEngine([1, 0])
    with pytest.raises(ValueError, match="match the number of vectors"):
        LinearCombinations.compute([1, 2], [v1])

def test_span_membership_inside():
    v1 = VectorEngine([1, 0])
    v2 = VectorEngine([0, 1])
    target = VectorEngine([5, 6])
    in_span, _ = SpanAnalyzer.is_in_span(target, [v1, v2])
    assert in_span is True

def test_span_membership_outside():
    v1 = VectorEngine([1, 2, 3])
    v2 = VectorEngine([2, 4, 6])
    target = VectorEngine([1, 0, 0]) # Not in the 1D span of [1,2,3]
    in_span, _ = SpanAnalyzer.is_in_span(target, [v1, v2])
    assert in_span is False

def test_linear_independence_dependent():
    v1 = VectorEngine([1, 2])
    v2 = VectorEngine([2, 4])
    report = LinearIndependenceAnalyzer.analyze([v1, v2])
    assert report["is_independent"] is False
    assert report["rank"] == 1

def test_linear_independence_exceeds_dimension():
    v1 = VectorEngine([1, 0])
    v2 = VectorEngine([0, 1])
    v3 = VectorEngine([1, 1])
    report = LinearIndependenceAnalyzer.analyze([v1, v2, v3])
    # 3 vectors in R^2 must be dependent
    assert report["is_independent"] is False

def test_subspace_extraction_dependent_cols():
    A = MatrixEngine([[1, 2], [3, 6]])
    # RREF of A has 1 pivot, so column space has 1 vector, null space has 1 vector
    col_basis = SubspaceAnalyzer.get_column_space(A)
    null_basis = SubspaceAnalyzer.get_null_space(A)
    
    assert len(col_basis) == 1
    assert len(null_basis) == 1

def test_axiom_checker():
    u = VectorEngine([1, 2])
    v = VectorEngine([3, 4])
    w = VectorEngine([0, -1])
    checks = AxiomExplorer.explore_axioms(u, v, w, 2.0, 3.0)
    
    assert checks["Commutativity ($u + v = v + u$)"] is True
    assert checks["Associativity ($(u + v) + w = u + (v + w)$)"] is True
