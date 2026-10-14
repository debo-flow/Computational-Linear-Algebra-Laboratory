from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.systems.system_model import LinearSystem
from linear_algebra.systems.classification import SystemClassifier
from linear_algebra.systems.solvers import SystemSolvers
from linear_algebra.systems.verification import SystemVerification
from linear_algebra.vector_spaces.span_analysis import SpanAnalyzer

class CoordinateSystem:
    @staticmethod
    def get_coordinates(basis_vectors, target_vector, tol=1e-9):
        """Computes coordinates [v]_B of a target vector relative to an ordered basis."""
        P = SpanAnalyzer.assemble_matrix(basis_vectors)
        B = SpanAnalyzer.assemble_matrix([target_vector])
        
        sys = LinearSystem(P, B)
        report = SystemClassifier.classify(sys, tol)
        
        if report['solution_type'] == 'none':
            raise ValueError("Target vector is OUTSIDE the span of the provided basis.")
        if report['solution_type'] == 'infinite':
            raise ValueError("Basis vectors are dependent; coordinates are not unique.")
            
        coords = SystemSolvers.direct_solve(sys) if P.is_square() else SystemSolvers.least_squares(sys)[0]
        
        # Verification
        R, _, total_norm = SystemVerification.compute_residual(sys, coords)
        
        return coords, total_norm

    @staticmethod
    def change_of_basis_matrix(basis_B, basis_C, tol=1e-9):
        """
        Computes the change of coordinate matrix P_{C <- B}.
        [v]_C = P_{C <- B} [v]_B
        Solved via P_C * X = P_B
        """
        P_B = SpanAnalyzer.assemble_matrix(basis_B)
        P_C = SpanAnalyzer.assemble_matrix(basis_C)
        
        if P_B.rows != P_C.rows:
            raise ValueError("Bases must inhabit the same ambient vector space dimension.")
            
        sys = LinearSystem(P_C, P_B)
        report = SystemClassifier.classify(sys, tol)
        
        if report['solution_type'] == 'none':
            raise ValueError("Basis B is not entirely contained in the span of Basis C.")
            
        if P_C.is_square():
            X = SystemSolvers.direct_solve(sys)
        else:
            X, _, _, _ = SystemSolvers.least_squares(sys)
            
        return X
