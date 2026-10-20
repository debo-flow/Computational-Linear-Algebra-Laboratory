import numpy as np
from linear_algebra.matrix.matrix_engine import MatrixEngine

class FactorizationDiagnostics:
    @staticmethod
    def analyze_system(A: MatrixEngine, b: MatrixEngine, x: MatrixEngine):
        """
        Calculates Absolute and Relative Forward Residuals for solved systems.
        """
        A_mat = np.array(A.matrix, dtype=complex if np.iscomplexobj(A.matrix) else float)
        b_mat = np.array(b.matrix, dtype=complex if np.iscomplexobj(b.matrix) else float)
        x_mat = np.array(x.matrix, dtype=complex if np.iscomplexobj(x.matrix) else float)
        
        fitted = A_mat @ x_mat
        residual = b_mat - fitted
        
        res_norm = np.linalg.norm(residual, 'fro')
        b_norm = max(np.linalg.norm(b_mat, 'fro'), 1e-15)
        
        cond_A = np.linalg.cond(A_mat) if A.rows == A.cols else float('inf')
        
        return {
            "abs_residual": res_norm,
            "rel_residual": res_norm / b_norm,
            "condition_number": cond_A,
            "warning": "Ill-conditioned system. Small residuals do not guarantee high solution accuracy." if cond_A > 1e10 else None
        }
