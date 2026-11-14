import numpy as np
import scipy.sparse.linalg as spla

class SparseEigenEngine:
    @staticmethod
    def compute_eigs(A_sparse, k=6, which='LM', tol=1e-5):
        """
        Computes k eigenvalues of a large sparse matrix using ARPACK.
        which options: 
        'LM' : Largest Magnitude
        'SM' : Smallest Magnitude
        'LR' : Largest Real part
        'SR' : Smallest Real part
        """
        m, n = A_sparse.shape
        if m != n:
            raise ValueError("Eigenvalue computation requires a square matrix.")
            
        if k >= n:
            raise ValueError(f"k ({k}) must be strictly less than matrix dimension n ({n}) for ARPACK sparse solvers. If you need all eigenvalues, use dense algorithms.")
            
        try:
            # eigs is for general matrices
            eigvals, eigvecs = spla.eigs(A_sparse, k=k, which=which, tol=tol)
            
            # Verify residuals
            residuals = []
            for i in range(k):
                v = eigvecs[:, i]
                val = eigvals[i]
                res = np.linalg.norm(A_sparse @ v - val * v)
                residuals.append(res)
                
            return {
                "eigenvalues": eigvals,
                "eigenvectors": eigvecs,
                "residuals": residuals,
                "status": "Converged"
            }
        except spla.ArpackNoConvergence as e:
            # Extract partial results if convergence fails
            return {
                "eigenvalues": e.eigenvalues,
                "eigenvectors": e.eigenvectors,
                "residuals": [],
                "status": f"ARPACK Failed to converge fully. {e}"
            }
