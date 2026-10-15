import pytest
import numpy as np
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.vectors.vector_engine import VectorEngine
from linear_algebra.transformations.geometric import GeometricTransformations
from linear_algebra.transformations.analysis import TransformationAnalyzer
from linear_algebra.transformations.basis_representation import TransformationBasis
from linear_algebra.transformations.composition import TransformationComposition

def test_rotation_properties():
    # 90 degrees rotation
    R = GeometricTransformations.rotation_2d(90, is_degrees=True)
    # Check R^T R = I (Orthogonality)
    R_T = MatrixEngine(R.matrix.T)
    I_approx = R_T.matrix @ R.matrix
    assert np.allclose(I_approx, np.eye(2))
    # Determinant scaling factor must be 1
    report = TransformationAnalyzer.analyze_properties(R)
    assert np.isclose(report["volume_scaling_factor"], 1.0)

def test_transformation_analysis():
    # Rectangular matrix: R^3 -> R^2
    A = MatrixEngine([[1, 2, 3], [4, 5, 6]])
    report = TransformationAnalyzer.analyze_properties(A)
    assert report["dim_domain"] == 3
    assert report["dim_codomain"] == 2
    assert report["rank"] == 2
    assert report["nullity"] == 1
    assert report["is_surjective"] is True
    assert report["is_injective"] is False
    # Volume scaling undefined for rectangular
    assert report["volume_scaling_factor"] is None

def test_basis_representation():
    # A maps (x,y) -> (x+y, y)
    A = MatrixEngine([[1, 1], [0, 1]])
    # Basis B: e1, e2
    B = [VectorEngine([1, 0]), VectorEngine([0, 1])]
    # Basis C: e2, e1 (Swapped)
    C = [VectorEngine([0, 1]), VectorEngine([1, 0])]
    
    # [T]_{C<-B} should be swapped rows of A
    T_CB = TransformationBasis.compute_matrix_in_bases(A, B, C)
    assert np.allclose(T_CB.matrix, [[0, 1], [1, 1]])

def test_composition_order():
    R = GeometricTransformations.rotation_2d(90, is_degrees=True)
    S = GeometricTransformations.scaling([2, 1]) # Scale X by 2
    
    # Compose S(R(x)) -> B*A -> S*R
    SR = TransformationComposition.compose(S, R)
    expected = np.array([[2, 0], [0, 1]]) @ np.array([[0, -1], [1, 0]])
    assert np.allclose(SR.matrix, expected)

def test_inverse_singular():
    # Singular matrix
    A = MatrixEngine([[1, 1], [1, 1]])
    with pytest.raises(ValueError, match="singular \\(non-bijective\\)"):
        TransformationComposition.inverse_transformation(A)
