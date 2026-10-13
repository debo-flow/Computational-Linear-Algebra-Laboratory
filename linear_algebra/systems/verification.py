import numpy as np
import sympy as sp
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.matrix.operations import MatrixOperations
from .system_model import LinearSystem

class SystemVerification:
    @staticmethod
    def compute_residual(system: LinearSystem, X: MatrixEngine):
        """Computes r = AX - B and its norms."""
        AX = MatrixOperations.multiply(system.A, X)
        if system.is_symbolic:
            res_matrix = sp.simplify(sp.Matrix(AX.matrix) - sp.Matrix(system.B.matrix))
            return MatrixEngine(res_matrix, True), [0.0], 0.0
        
        diff = AX.matrix - system.B.matrix
        col_norms = np.linalg.norm(diff, axis=0).tolist()
        total_norm = np.linalg.norm(diff)
        return MatrixEngine(diff), col_norms, total_norm
