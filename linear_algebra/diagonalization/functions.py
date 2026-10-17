import numpy as np
import sympy as sp
import scipy.linalg as spla
from linear_algebra.matrix.matrix_engine import MatrixEngine

class MatrixFunctions:
    @staticmethod
    def matrix_power_diag(P: MatrixEngine, D: MatrixEngine, P_inv: MatrixEngine, n: int):
        """Computes A^n = P D^n P^(-1)."""
        if n < 0:
            raise ValueError("Negative powers require strict invertibility checks; use integer n >= 0.")
        if P.is_symbolic:
            D_n = sp.Matrix(D.matrix)**n
            A_n = sp.Matrix(P.matrix) @ D_n @ sp.Matrix(P_inv.matrix)
            return MatrixEngine(sp.simplify(A_n), True)
        else:
            D_n = np.diag(np.diag(D.matrix)**n)
            A_n = P.matrix @ D_n @ P_inv.matrix
            return MatrixEngine(A_n)

    @staticmethod
    def matrix_exp_diag(P: MatrixEngine, D: MatrixEngine, P_inv: MatrixEngine):
        """Computes e^A = P e^D P^(-1)."""
        if P.is_symbolic:
            D_exp = sp.diag(*[sp.exp(D.matrix[i, i]) for i in range(D.rows)])
            A_exp = sp.Matrix(P.matrix) @ D_exp @ sp.Matrix(P_inv.matrix)
            return MatrixEngine(sp.simplify(A_exp), True)
        else:
            D_exp = np.diag(np.exp(np.diag(D.matrix)))
            A_exp = P.matrix @ D_exp @ P_inv.matrix
            return MatrixEngine(A_exp)
