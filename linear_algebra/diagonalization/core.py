import numpy as np
import sympy as sp
from linear_algebra.matrix.matrix_engine import MatrixEngine

class DiagonalizationEngine:
    @staticmethod
    def diagonalize(A: MatrixEngine, tol=1e-9):
        """
        Diagonalizes a square matrix A = P D P^(-1).
        Computes the reconstruction residual and checks condition number of P.
        """
        if not A.is_square():
            raise ValueError("Only square matrices can be mathematically diagonalized.")

        report = {
            "is_diagonalizable": False,
            "P": None,
            "D": None,
            "P_inv": None,
            "residual": None,
            "condition_number": None,
            "message": ""
        }

        if A.is_symbolic:
            sym_A = sp.Matrix(A.matrix)
            try:
                P, D = sym_A.diagonalize()
                report["is_diagonalizable"] = True
                report["P"] = MatrixEngine(P, True)
                report["D"] = MatrixEngine(D, True)
                report["P_inv"] = MatrixEngine(P.inv(), True)
                report["residual"] = 0.0
                report["message"] = "Matrix is strictly diagonalizable over the exact symbolic field."
            except sp.MatrixError:
                report["is_diagonalizable"] = False
                report["message"] = "Matrix is defective. It does not possess a complete basis of eigenvectors."
        else:
            vals, vecs = np.linalg.eig(A.matrix)
            P_mat = vecs
            D_mat = np.diag(vals)

            # Condition number check to establish numerical diagonalizability
            cond_P = np.linalg.cond(P_mat)
            report["condition_number"] = cond_P

            if cond_P > (1 / tol):
                report["is_diagonalizable"] = False
                report["message"] = f"Matrix is numerically defective or ill-conditioned. Cond(P) = {cond_P:.2e}."
            else:
                P_inv = np.linalg.inv(P_mat)
                A_recon = P_mat @ D_mat @ P_inv
                
                norm_A = np.linalg.norm(A.matrix, 'fro')
                eps = 1e-15
                residual = np.linalg.norm(A.matrix - A_recon, 'fro') / max(norm_A, eps)

                report["is_diagonalizable"] = True
                report["P"] = MatrixEngine(P_mat)
                report["D"] = MatrixEngine(D_mat)
                report["P_inv"] = MatrixEngine(P_inv)
                report["residual"] = residual
                report["message"] = f"Numerically diagonalizable. Normalized reconstruction residual: {residual:.2e}."

        return report
