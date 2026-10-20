import numpy as np
import scipy.linalg as spla
from linear_algebra.matrix.matrix_engine import MatrixEngine

class LUEngine:
    @staticmethod
    def compute_partial_pivot(A: MatrixEngine, tol=1e-9):
        """
        Computes LU Decomposition with Partial Pivoting.
        Mathematical Convention: P A = L U
        """
        mat = np.array(A.matrix, dtype=complex if np.iscomplexobj(A.matrix) else float)
        if not A.is_square():
            raise ValueError("LU Decomposition requires a square matrix.")
            
        # SciPy returns P_sp, L, U such that A = P_sp @ L @ U.
        # To satisfy P A = L U, we take P = P_sp.T (since permutation matrices are orthogonal).
        P_sp, L, U = spla.lu(mat)
        P = P_sp.T
        
        # Reconstruction: A ≈ P^T L U
        A_recon = P.T @ L @ U
        recon_err = np.linalg.norm(mat - A_recon, 'fro') / max(np.linalg.norm(mat, 'fro'), 1e-15)
        
        return {
            "P": MatrixEngine(P),
            "L": MatrixEngine(L),
            "U": MatrixEngine(U),
            "reconstruction_error": recon_err
        }

    @staticmethod
    def compute_no_pivot(A: MatrixEngine, tol=1e-9):
        """
        Educational: Computes LU Decomposition without pivoting.
        Will fail mathematically on zero or near-zero pivots.
        """
        mat = np.array(A.matrix, dtype=complex if np.iscomplexobj(A.matrix) else float)
        if not A.is_square():
            raise ValueError("LU Decomposition requires a square matrix.")
            
        n = mat.shape[0]
        L = np.eye(n, dtype=mat.dtype)
        U = mat.copy()
        
        steps = []
        for i in range(n):
            if abs(U[i, i]) < tol:
                raise ValueError(f"Zero or near-zero pivot ({abs(U[i,i]):.2e}) encountered at row {i+1}. LU without pivoting fails. Partial pivoting is required.")
            
            for j in range(i+1, n):
                factor = U[j, i] / U[i, i]
                L[j, i] = factor
                U[j, i:] -= factor * U[i, i:]
                
                # Educational capture
                steps.append({
                    "description": f"Eliminate row {j+1} using row {i+1} (multiplier = {factor:.4g})",
                    "L_current": MatrixEngine(L.copy()),
                    "U_current": MatrixEngine(U.copy())
                })
                
        A_recon = L @ U
        recon_err = np.linalg.norm(mat - A_recon, 'fro') / max(np.linalg.norm(mat, 'fro'), 1e-15)
        
        return {
            "L": MatrixEngine(L),
            "U": MatrixEngine(U),
            "reconstruction_error": recon_err,
            "steps": steps
        }
