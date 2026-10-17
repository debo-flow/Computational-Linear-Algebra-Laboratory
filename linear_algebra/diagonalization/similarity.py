import numpy as np
import sympy as sp
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.matrix.operations import MatrixOperations

class SimilarityAnalyzer:
    @staticmethod
    def transform(A: MatrixEngine, P: MatrixEngine, tol=1e-9):
        """
        Computes the similarity transformation B = P^(-1) A P.
        """
        if not A.is_square() or not P.is_square() or A.shape != P.shape:
            raise ValueError("A and P must be square matrices of the exact same dimension.")
        
        P_inv = MatrixOperations.inverse(P, tol)
        
        if A.is_symbolic:
            B_sym = sp.Matrix(P_inv.matrix) @ sp.Matrix(A.matrix) @ sp.Matrix(P.matrix)
            return MatrixEngine(sp.simplify(B_sym), True), P_inv
        else:
            B_num = P_inv.matrix @ A.matrix @ P.matrix
            return MatrixEngine(B_num), P_inv
