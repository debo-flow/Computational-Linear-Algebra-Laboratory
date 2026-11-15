import numpy as np
import scipy.sparse.linalg as spla

class ResidualHistory:
    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.history = []
        self.b_norm = np.linalg.norm(b)
        if self.b_norm < 1e-15:
            self.b_norm = 1.0
            
    def __call__(self, xk):
        # Handle GMRES passing residual directly vs CG passing state vector
        if np.isscalar(xk):
            self.history.append(float(xk))
        else:
            res = np.linalg.norm(self.b - self.A @ xk)
            self.history.append(res / self.b_norm)

class IterativeSolvers:
    @staticmethod
    def cg(A, b, x0=None, tol=1e-5, maxiter=1000, M=None):
        """Conjugate Gradient (Requires Symmetric Positive-Definite)."""
        hist = ResidualHistory(A, b)
        x, info = spla.cg(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M, callback=hist)
        return x, info, hist.history

    @staticmethod
    def gmres(A, b, x0=None, tol=1e-5, maxiter=1000, restart=None, M=None):
        """Generalized Minimal Residual (Robust for nonsymmetric matrices)."""
        hist = ResidualHistory(A, b)
        x, info = spla.gmres(A, b, x0=x0, tol=tol, maxiter=maxiter, restart=restart, M=M, callback=hist)
        return x, info, hist.history

    @staticmethod
    def bicgstab(A, b, x0=None, tol=1e-5, maxiter=1000, M=None):
        """Biconjugate Gradient Stabilized."""
        hist = ResidualHistory(A, b)
        x, info = spla.bicgstab(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M, callback=hist)
        return x, info, hist.history

    @staticmethod
    def minres(A, b, x0=None, tol=1e-5, maxiter=1000, M=None):
        """Minimum Residual (Requires Symmetric Indefinite/Definite)."""
        hist = ResidualHistory(A, b)
        x, info = spla.minres(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M, callback=hist)
        return x, info, hist.history
