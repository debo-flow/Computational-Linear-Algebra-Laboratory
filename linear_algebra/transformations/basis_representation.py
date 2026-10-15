from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.matrix.operations import MatrixOperations
from linear_algebra.systems.system_model import LinearSystem
from linear_algebra.systems.solvers import SystemSolvers
from linear_algebra.vector_spaces.span_analysis import SpanAnalyzer

class TransformationBasis:
    @staticmethod
    def compute_matrix_in_bases(A: MatrixEngine, basis_B, basis_C):
        """
        Computes [T]_{C <- B} for a linear transformation T(x) = Ax.
        Formula: [T]_{C <- B} = P_C^{-1} A P_B
        """
        P_B = SpanAnalyzer.assemble_matrix(basis_B)
        P_C = SpanAnalyzer.assemble_matrix(basis_C)
        
        if A.cols != P_B.rows:
            raise ValueError("Domain basis dimensions do not match transformation domain.")
        if A.rows != P_C.rows:
            raise ValueError("Codomain basis dimensions do not match transformation codomain.")
            
        # Transform basis B using standard matrix A
        AP_B = MatrixOperations.multiply(A, P_B)
        
        # Solve P_C * X = A * P_B
        sys = LinearSystem(P_C, AP_B)
        if P_C.is_square():
            return SystemSolvers.direct_solve(sys)
        else:
            # Overdetermined systems require exact least-squares solutions if C truly spans the codomain
            X, residuals, _, _ = SystemSolvers.least_squares(sys)
            if len(residuals) > 0 and residuals[0] > 1e-7:
                raise ValueError("Transformation maps outside the span of codomain Basis C.")
            return X
