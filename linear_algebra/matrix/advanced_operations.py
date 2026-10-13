import numpy as np
import scipy.linalg as spla
import sympy as sp
from .matrix_engine import MatrixEngine
from .operations import MatrixOperations

class AdvancedMatrixOperations:
    @staticmethod
    def hadamard_product(A: MatrixEngine, B: MatrixEngine):
        if A.shape != B.shape:
            raise ValueError(f"Hadamard product undefined: Dimensions {A.shape} and {B.shape} must match.")
        if A.is_symbolic or B.is_symbolic:
            return MatrixEngine(sp.matrix_multiply_elementwise(sp.Matrix(A.matrix), sp.Matrix(B.matrix)), True)
        return MatrixEngine(np.multiply(A.matrix, B.matrix))

    @staticmethod
    def elementwise_division(A: MatrixEngine, B: MatrixEngine, tol=1e-12):
        if A.shape != B.shape:
            raise ValueError(f"Element-wise division undefined: Dimensions {A.shape} and {B.shape} must match.")
        
        if B.is_symbolic:
            # SymPy handles division by zero symbolically, but we check for explicit zeros
            if any(val == 0 for val in sp.Matrix(B.matrix)):
                raise ValueError("Element-wise division undefined: Division by zero detected in Matrix B.")
            div_mat = sp.Matrix(A.matrix).multiply_elementwise(sp.Matrix(B.matrix).applyfunc(lambda x: 1/x))
            return MatrixEngine(div_mat, True)
        else:
            if np.any(np.abs(B.matrix) < tol):
                raise ValueError("Element-wise division undefined: Division by near-zero detected in Matrix B.")
            return MatrixEngine(np.divide(A.matrix, B.matrix))

    @staticmethod
    def kronecker_product(A: MatrixEngine, B: MatrixEngine):
        if A.is_symbolic or B.is_symbolic:
            # SymPy lacks a direct Kronecker product method on Matrices that works flawlessly in all versions, 
            # so we build it block by block or use tensorproduct
            from sympy.physics.quantum import TensorProduct
            res = TensorProduct(sp.Matrix(A.matrix), sp.Matrix(B.matrix))
            return MatrixEngine(res, True)
        return MatrixEngine(np.kron(A.matrix, B.matrix))

    @staticmethod
    def matrix_power(A: MatrixEngine, n: int):
        if not A.is_square():
            raise ValueError("Matrix power undefined: Matrix must be square.")
        if n < 0:
            # Requires inverse
            A_inv = MatrixOperations.inverse(A)
            if A.is_symbolic:
                return MatrixEngine(sp.Matrix(A_inv.matrix)**abs(n), True)
            return MatrixEngine(np.linalg.matrix_power(A_inv.matrix, abs(n)))
        
        if A.is_symbolic:
            return MatrixEngine(sp.Matrix(A.matrix)**n, True)
        return MatrixEngine(np.linalg.matrix_power(A.matrix, n))

    @staticmethod
    def scalar_shift(A: MatrixEngine, lambda_val):
        """Computes A + λI"""
        if not A.is_square():
            raise ValueError("Scalar shift (A + λI) requires a square matrix.")
        I = MatrixEngine.generate_identity(A.rows, A.is_symbolic)
        if A.is_symbolic:
            shift = sp.Matrix(I.matrix) * lambda_val
            return MatrixEngine(sp.Matrix(A.matrix) + shift, True)
        return MatrixEngine(A.matrix + lambda_val * I.matrix)

    @staticmethod
    def matrix_exponential(A: MatrixEngine):
        """Computes exp(A) using Padé approximation for numeric, exp() for symbolic"""
        if not A.is_square():
            raise ValueError("Matrix exponential undefined: Matrix must be square.")
        if A.is_symbolic:
            try:
                # SymPy exp can be very slow for n > 3
                return MatrixEngine(sp.Matrix(A.matrix).exp(), True)
            except Exception as e:
                raise ValueError(f"Symbolic matrix exponential failed or too complex: {e}")
        return MatrixEngine(spla.expm(A.matrix))
