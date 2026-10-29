import numpy as np
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.vectors.vector_engine import VectorEngine

class GeneralizedRayleighQuotient:
    @staticmethod
    def compute(A: MatrixEngine, B: MatrixEngine, x: VectorEngine, tol=1e-9):
        """
        Computes the Generalized Rayleigh Quotient R(x) = (x^H A x) / (x^H B x)
        """
        x_arr = np.array(x.vector).reshape(-1, 1)
        mat_A = np.array(A.matrix)
        mat_B = np.array(B.matrix)
        
        num = x_arr.conj().T @ mat_A @ x_arr
        den = x_arr.conj().T @ mat_B @ x_arr
        
        num_val = num[0, 0]
        den_val = den[0, 0]
        
        if abs(den_val) < tol:
            raise ValueError("Denominator (x^H B x) is zero or numerically negligible. Generalized Rayleigh Quotient undefined.")
            
        # Mathematically real if A, B are Hermitian
        is_hermitian_A = np.allclose(mat_A, mat_A.conj().T, atol=tol)
        is_hermitian_B = np.allclose(mat_B, mat_B.conj().T, atol=tol)
        
        q_val = num_val / den_val
        
        if is_hermitian_A and is_hermitian_B:
            q_val = np.real(q_val)
            
        return q_val
