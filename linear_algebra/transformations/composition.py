from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.matrix.operations import MatrixOperations
from linear_algebra.systems.solvers import SystemSolvers
from linear_algebra.systems.system_model import LinearSystem
from linear_algebra.matrix.validation import ValidationEngine

class TransformationComposition:
    @staticmethod
    def compose(A: MatrixEngine, B: MatrixEngine):
        """
        Composes S(T(x)) where T has matrix B and S has matrix A.
        Resulting matrix is A @ B.
        """
        return MatrixOperations.multiply(A, B)

    @staticmethod
    def inverse_transformation(A: MatrixEngine, tol=1e-9):
        """Computes the inverse linear transformation if bijective."""
        if not A.is_square():
            raise ValueError("Transformation is not an endomorphism (not square); standard inverse undefined.")
            
        is_val, res, _ = ValidationEngine.verify_inverse(A, tol)
        if not is_val:
            raise ValueError(f"Transformation is singular (non-bijective). Inverse does not exist.")
            
        return MatrixOperations.inverse(A, tol)
