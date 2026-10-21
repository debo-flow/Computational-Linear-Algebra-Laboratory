import numpy as np
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.vectors.vector_engine import VectorEngine
from linear_algebra.orthogonality.inner_product import InnerProductEngine

class QuadraticForms:
    @staticmethod
    def evaluate(A: MatrixEngine, x: VectorEngine):
        """
        Evaluates q(x) = x^H A x.
        Mathematically, for Hermitian A, this is strictly real.
        """
        x_arr = np.array(x.vector).reshape(-1, 1)
        mat = np.array(A.matrix)
        
        qx = x_arr.conj().T @ mat @ x_arr
        return np.real(qx[0, 0]) # Extract strictly real scalar

    @staticmethod
    def rayleigh_quotient(A: MatrixEngine, x: VectorEngine, tol=1e-9):
        """
        Evaluates R_A(x) = (x^H A x) / (x^H x).
        Bounded by lambda_min <= R_A(x) <= lambda_max.
        """
        norm_sq = InnerProductEngine.norm(x)**2
        if norm_sq < tol:
            raise ValueError("Rayleigh quotient is mathematically undefined for the zero vector.")
            
        qx = QuadraticForms.evaluate(A, x)
        return qx / norm_sq

    @staticmethod
    def gradient_real(A: MatrixEngine, x: VectorEngine):
        """
        Computes ∇q(x) = 2Ax.
        Assumes strictly Real Symmetric matrix A.
        """
        if np.iscomplexobj(A.matrix) or np.iscomplexobj(x.vector):
            raise ValueError("Gradient calculation natively supported only for real symmetric matrices in this module.")
        
        x_arr = np.array(x.vector).reshape(-1, 1)
        mat = np.array(A.matrix)
        
        grad = 2 * (mat @ x_arr)
        return VectorEngine(grad.flatten())
