import numpy as np
import sympy as sp
from .matrix_engine import MatrixEngine
from .operations import MatrixOperations

class DeterminantProperties:
    @staticmethod
    def verify_transpose(A: MatrixEngine, tol=1e-9):
        """Verifies det(A^T) = det(A)"""
        if not A.is_square():
            raise ValueError("Matrix must be square.")
        det_A = MatrixOperations.determinant(A)
        A_T = MatrixEngine(A.matrix.T, A.is_symbolic)
        det_AT = MatrixOperations.determinant(A_T)
        
        if A.is_symbolic:
            return sp.simplify(det_A - det_AT) == 0, 0.0, det_A, det_AT
        
        diff = abs(det_A - det_AT)
        return diff < tol, diff, det_A, det_AT

    @staticmethod
    def verify_multiplication(A: MatrixEngine, B: MatrixEngine, tol=1e-9):
        """Verifies det(AB) = det(A)det(B)"""
        if not A.is_square() or not B.is_square() or A.shape != B.shape:
            raise ValueError("Matrices must be square and of identical dimension.")
            
        AB = MatrixOperations.multiply(A, B)
        det_AB = MatrixOperations.determinant(AB)
        det_A = MatrixOperations.determinant(A)
        det_B = MatrixOperations.determinant(B)
        
        if A.is_symbolic:
            LHS = sp.simplify(det_AB)
            RHS = sp.simplify(det_A * det_B)
            return sp.simplify(LHS - RHS) == 0, 0.0, LHS, RHS
            
        LHS = det_AB
        RHS = det_A * det_B
        diff = abs(LHS - RHS)
        return diff < tol, diff, LHS, RHS
