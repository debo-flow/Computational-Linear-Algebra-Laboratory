import numpy as np
import scipy.linalg as spla
from linear_algebra.matrix.matrix_engine import MatrixEngine
from .core import QREngine

class QRLeastSquares:
    @staticmethod
    def solve(A: MatrixEngine, B: MatrixEngine, tol=1e-9):
        """
        Solves Overdetermined AX ≈ B via Reduced QR.
        Rx = Q^H B. Handles multiple RHS.
        """
        mat_A = np.array(A.matrix, dtype=complex if np.iscomplexobj(A.matrix) else float)
        mat_B = np.array(B.matrix, dtype=complex if np.iscomplexobj(B.matrix) else float)
        
        rank = np.linalg.matrix_rank(mat_A, tol=tol)
        if rank < A.cols:
            return {"status": "rank_deficient", "rank": rank}
            
        qr_data = QREngine.compute_library_qr(A, mode='reduced', tol=tol)
        Q, R = np.array(qr_data['Q'].matrix), np.array(qr_data['R'].matrix)
        
        # Solve R X = Q^H B
        Q_H_B = Q.conj().T @ mat_B
        X = spla.solve_triangular(R, Q_H_B)
        
        # Residuals
        fitted = mat_A @ X
        residuals = mat_B - fitted
        res_norms = np.linalg.norm(residuals, axis=0)
        overall_res = np.linalg.norm(residuals, 'fro')
        
        return {
            "status": "success",
            "X": MatrixEngine(X),
            "fitted": MatrixEngine(fitted),
            "residuals": MatrixEngine(residuals),
            "residual_norms": res_norms,
            "overall_residual": overall_res,
            "rank": rank
        }

    @staticmethod
    def solve_weighted(A: MatrixEngine, B: MatrixEngine, weights: list, tol=1e-9):
        """
        Solves Weighted Least Squares: min (Ax-b)^H W (Ax-b).
        Transforms A_w = W^(1/2) A, B_w = W^(1/2) B.
        """
        if any(w < 0 for w in weights):
            raise ValueError("Weights must be strictly non-negative positive values.")
        if len(weights) != A.rows:
            raise ValueError("Number of weights must equal the number of equations (rows).")
            
        W_half = np.diag(np.sqrt(weights))
        A_w = MatrixEngine(W_half @ np.array(A.matrix, dtype=float))
        B_w = MatrixEngine(W_half @ np.array(B.matrix, dtype=float))
        
        return QRLeastSquares.solve(A_w, B_w, tol)
