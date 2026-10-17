import numpy as np
from linear_algebra.matrix.matrix_engine import MatrixEngine
from .core import SVDEngine

class LowRankApproximation:
    @staticmethod
    def truncate(A: MatrixEngine, k: int, tol=1e-9):
        """
        Computes the optimal rank-k approximation A_k = U_k Σ_k V_k^H
        Returns reconstruction, error metrics, and retained energy.
        """
        svd_data = SVDEngine.compute(A, full_matrices=False, tol=tol)
        U, S, Vh = svd_data["U"].matrix, svd_data["S"], svd_data["Vh"].matrix
        
        # Bound k by the number of singular values
        k = max(1, min(k, len(S)))
        
        U_k = U[:, :k]
        S_k = S[:k]
        Vh_k = Vh[:k, :]
        
        A_k = U_k @ np.diag(S_k) @ Vh_k
        if np.allclose(A_k.imag, 0):
            A_k = A_k.real
            
        # Error analysis (Eckart-Young Theorem validations)
        total_energy = np.sum(S**2)
        retained_energy = np.sum(S_k**2)
        energy_ratio = retained_energy / total_energy if total_energy > 0 else 1.0
        
        mat = np.array(A.matrix, dtype=float) if not np.iscomplexobj(A.matrix) else np.array(A.matrix, dtype=complex)
        diff = mat - A_k
        abs_error_frob = np.linalg.norm(diff, 'fro')
        
        # Theoretical Eckart-Young errors
        theoretical_frob_error = np.sqrt(np.sum(S[k:]**2)) if k < len(S) else 0.0
        theoretical_spectral_error = S[k] if k < len(S) else 0.0
        
        rel_error = abs_error_frob / max(np.linalg.norm(mat, 'fro'), 1e-15)
        
        return {
            "A_k": MatrixEngine(A_k),
            "U_k": MatrixEngine(U_k),
            "S_k": S_k,
            "Vh_k": MatrixEngine(Vh_k),
            "rank_k": k,
            "abs_error_frob": abs_error_frob,
            "rel_error_frob": rel_error,
            "theoretical_frob_error": theoretical_frob_error,
            "theoretical_spectral_error": theoretical_spectral_error,
            "retained_energy_ratio": energy_ratio
        }
