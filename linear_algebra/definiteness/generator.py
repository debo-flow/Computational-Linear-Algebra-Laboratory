import numpy as np
from linear_algebra.matrix.matrix_engine import MatrixEngine

class DefiniteMatrixGenerator:
    @staticmethod
    def generate(n: int, target_type: str, is_complex=False, seed=None):
        """
        Generates matrices with known definiteness properties.
        A = B^H B + alpha I
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Generate random B
        if is_complex:
            B = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        else:
            B = np.random.randn(n, n)
            
        B_HB = B.conj().T @ B
        
        if target_type == "PD":
            mat = B_HB + 0.5 * np.eye(n) # Strictly positive
        elif target_type == "PSD":
            # B needs to be rank deficient to ensure an exact 0 eigenvalue
            B_low = np.random.randn(n-1, n)
            mat = B_low.conj().T @ B_low
        elif target_type == "ND":
            mat = -(B_HB + 0.5 * np.eye(n))
        elif target_type == "NSD":
            B_low = np.random.randn(n-1, n)
            mat = -(B_low.conj().T @ B_low)
        elif target_type == "Indefinite":
            mat = B + B.conj().T # Random symmetric generally indefinite
        else:
            raise ValueError("Unknown target type.")
            
        return MatrixEngine(mat)
