import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

class StationarySolvers:
    @staticmethod
    def _prepare_sparse(A):
        if not sps.issparse(A):
            return sps.csr_matrix(A)
        return A.tocsr()

    @staticmethod
    def jacobi(A, b, x0=None, tol=1e-5, maxiter=1000):
        """Jacobi Iteration: x_{k+1} = D^-1 (b - R x_k)"""
        A = StationarySolvers._prepare_sparse(A)
        n = A.shape[0]
        x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
        
        D = A.diagonal()
        if np.any(np.abs(D) < 1e-15):
            raise ValueError("Zero on diagonal detected. Jacobi iteration cannot proceed.")
            
        R = A - sps.diags(D)
        b_norm = max(np.linalg.norm(b), 1e-15)
        history = []
        
        for _ in range(maxiter):
            x_new = (b - R @ x) / D
            res = np.linalg.norm(b - A @ x_new)
            rel_res = res / b_norm
            history.append(rel_res)
            x = x_new
            if rel_res < tol:
                return x, 0, history
        return x, 1, history

    @staticmethod
    def gauss_seidel(A, b, x0=None, tol=1e-5, maxiter=1000):
        """Gauss-Seidel Iteration: L x_{k+1} = b - U x_k"""
        A = StationarySolvers._prepare_sparse(A)
        n = A.shape[0]
        x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
        
        L = sps.tril(A, format='csr')
        U = sps.triu(A, k=1, format='csr')
        
        b_norm = max(np.linalg.norm(b), 1e-15)
        history = []
        
        for _ in range(maxiter):
            rhs = b - U @ x
            x_new = spla.spsolve_triangular(L, rhs, lower=True)
            res = np.linalg.norm(b - A @ x_new)
            rel_res = res / b_norm
            history.append(rel_res)
            x = x_new
            if rel_res < tol:
                return x, 0, history
        return x, 1, history
