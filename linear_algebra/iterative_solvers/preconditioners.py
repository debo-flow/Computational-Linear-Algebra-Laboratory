import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

class Preconditioners:
    @staticmethod
    def jacobi(A):
        """Diagonal (Jacobi) preconditioner M^-1 = D^-1."""
        d = A.diagonal().copy()
        d[d == 0] = 1.0  # Prevent division by zero safely
        
        def matvec(v):
            return v / d
            
        return spla.LinearOperator(A.shape, matvec=matvec)

    @staticmethod
    def ilu(A):
        """Incomplete LU (ILU) preconditioner."""
        if not sps.issparse(A):
            A = sps.csc_matrix(A)
            
        try:
            ilu_decomp = spla.spilu(A.tocsc(), drop_tol=1e-4)
            def matvec(v):
                return ilu_decomp.solve(v)
            return spla.LinearOperator(A.shape, matvec=matvec)
        except RuntimeError as e:
            raise ValueError(f"ILU Factorization failed (matrix may be singular or ill-conditioned): {e}")
