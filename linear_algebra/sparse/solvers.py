import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import time

class SparseSolverEngine:
    @staticmethod
    def direct_solve(A_sparse, b):
        """Solves Ax = b using SuperLU (Sparse Direct)."""
        if A_sparse.shape[0] != A_sparse.shape[1]:
            raise ValueError("Direct solver requires a square matrix.")
        
        # Convert to CSC for SuperLU efficiency
        A_csc = A_sparse.tocsc()
        
        start = time.perf_counter()
        x = spla.spsolve(A_csc, b)
        end = time.perf_counter()
        
        residual = np.linalg.norm(A_csc @ x - b)
        return x, residual, end - start

    @staticmethod
    def iterative_solve(A_sparse, b, method='gmres', tol=1e-5, maxiter=1000):
        """Solves Ax = b using Iterative Krylov methods (CG, GMRES)."""
        if A_sparse.shape[0] != A_sparse.shape[1]:
            raise ValueError("Iterative solvers require square matrices.")
            
        history = []
        def callback(pr_norm):
            history.append(pr_norm)
            
        start = time.perf_counter()
        if method == 'cg':
            # CG assumes Symmetric Positive Definite
            x, info = spla.cg(A_sparse, b, tol=tol, maxiter=maxiter, callback=callback)
        elif method == 'gmres':
            # GMRES handles non-symmetric
            x, info = spla.gmres(A_sparse, b, tol=tol, maxiter=maxiter, callback=callback)
        elif method == 'bicgstab':
            x, info = spla.bicgstab(A_sparse, b, tol=tol, maxiter=maxiter, callback=callback)
        else:
            raise ValueError("Unsupported iterative solver.")
        end = time.perf_counter()
        
        residual = np.linalg.norm(A_sparse @ x - b)
        
        return {
            "x": x,
            "status": "Converged" if info == 0 else f"Failed/Max iterations reached (info={info})",
            "residual_norm": residual,
            "time_sec": end - start,
            "history": history
        }
