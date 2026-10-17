import numpy as np
import sympy as sp
from linear_algebra.matrix.matrix_engine import MatrixEngine

class SVDEngine:
    @staticmethod
    def compute(A: MatrixEngine, full_matrices=True, tol=1e-9):
        """
        Computes the Singular Value Decomposition A = U Σ V^H.
        Since exact symbolic SVD is often mathematically intractable for arbitrary roots,
        we cast to numerical domains via numpy, which provides robust LAPACK routines.
        """
        mat = np.array(A.matrix, dtype=complex if np.iscomplexobj(A.matrix) else float)
        
        U, S, Vh = np.linalg.svd(mat, full_matrices=full_matrices)
        
        # Rank and Condition Number Analysis
        numerical_rank = np.sum(S > tol)
        
        if len(S) > 0 and S[0] > tol:
            if numerical_rank == min(mat.shape):
                condition_number = S[0] / S[-1]
            else:
                condition_number = float('inf') # Computationally infinite due to rank deficiency
        else:
            condition_number = 0.0 # Zero matrix
            
        return {
            "U": MatrixEngine(U),
            "S": S,  # 1D array of singular values
            "Vh": MatrixEngine(Vh),
            "rank": int(numerical_rank),
            "condition_number": condition_number,
            "max_sv": S[0] if len(S) > 0 else 0.0,
            "min_sv": S[-1] if len(S) > 0 else 0.0,
            "min_nonzero_sv": S[numerical_rank-1] if numerical_rank > 0 else 0.0
        }

    @staticmethod
    def reconstruct(U: MatrixEngine, S: np.ndarray, Vh: MatrixEngine):
        """
        Reconstructs the original matrix A = U Σ V^H.
        Automatically maps the 1D S array back into the correct rectangular Σ matrix shape.
        """
        m = U.rows
        n = Vh.cols
        Sigma = np.zeros((m, n), dtype=complex if np.iscomplexobj(U.matrix) or np.iscomplexobj(Vh.matrix) else float)
        
        k = len(S)
        Sigma[:k, :k] = np.diag(S)
        
        A_recon = U.matrix @ Sigma @ Vh.matrix
        # Clean up near-zero imaginary parts if the result should be strictly real
        if np.allclose(A_recon.imag, 0):
            A_recon = A_recon.real
            
        return MatrixEngine(A_recon)
