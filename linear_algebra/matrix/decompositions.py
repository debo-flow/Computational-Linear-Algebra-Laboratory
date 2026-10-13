import numpy as np
import scipy.linalg as spla
import sympy as sp
from .matrix_engine import MatrixEngine

class MatrixDecompositions:
    @staticmethod
    def lu_decomposition(A: MatrixEngine):
        """Computes PA = LU (Returns P, L, U)."""
        if A.is_symbolic:
            # SymPy LUdecomposition returns L, U, and a list of row swaps
            L, U, swaps = sp.Matrix(A.matrix).LUdecomposition()
            P = sp.eye(A.rows)
            # Reconstruct P matrix from swaps
            for (i, j) in swaps:
                P.row_swap(i, j)
            return MatrixEngine(P, True), MatrixEngine(L, True), MatrixEngine(U, True)
        
        # SciPy returns P, L, U such that A = P @ L @ U. 
        P, L, U = spla.lu(A.matrix)
        return MatrixEngine(P), MatrixEngine(L), MatrixEngine(U)

    @staticmethod
    def cholesky_decomposition(A: MatrixEngine):
        """Computes A = LL^T for symmetric positive-definite matrices."""
        if not A.is_square():
            raise ValueError("Cholesky decomposition requires a square matrix.")
        if not A.is_symmetric():
            raise ValueError("Cholesky decomposition requires a symmetric/Hermitian matrix.")
        
        try:
            if A.is_symbolic:
                L = sp.Matrix(A.matrix).cholesky()
                return MatrixEngine(L, True)
            
            L = np.linalg.cholesky(A.matrix)
            return MatrixEngine(L)
        except np.linalg.LinAlgError:
            raise ValueError("Matrix is not positive-definite. Cholesky decomposition undefined.")
        except ValueError as e:
            raise ValueError(f"Symbolic Cholesky failed (likely not positive-definite): {e}")
