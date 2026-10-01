import numpy as np
from .matrix_engine import MatrixEngine
from .operations import MatrixOperations

class ValidationEngine:
    @staticmethod
    def verify_inverse(A: MatrixEngine, tol=1e-9):
        """Verifies A * A_inv ≈ I"""
        try:
            A_inv = MatrixOperations.inverse(A)
            I_calc = MatrixOperations.multiply(A, A_inv)
            I_true = MatrixEngine.generate_identity(A.rows, A.is_symbolic)
            
            if A.is_symbolic:
                import sympy as sp
                diff = sp.simplify(I_calc.matrix - I_true.matrix)
                is_valid = diff.is_zero_matrix
                residual = 0.0
            else:
                diff = I_calc.matrix - I_true.matrix
                residual = np.linalg.norm(diff)
                is_valid = residual < tol
                
            return is_valid, residual, I_calc
        except Exception as e:
            return False, str(e), None

    @staticmethod
    def verify_transpose_addition(A: MatrixEngine, B: MatrixEngine, tol=1e-9):
        """Verifies (A + B)^T = A^T + B^T"""
        LHS = MatrixOperations.add(A, B)
        LHS_T = MatrixEngine(LHS.matrix.T, LHS.is_symbolic)
        
        A_T = MatrixEngine(A.matrix.T, A.is_symbolic)
        B_T = MatrixEngine(B.matrix.T, B.is_symbolic)
        RHS = MatrixOperations.add(A_T, B_T)
        
        if A.is_symbolic:
            diff = (LHS_T.matrix - RHS.matrix).simplify()
            return diff.is_zero_matrix, 0.0, LHS_T, RHS
        else:
            diff = LHS_T.matrix - RHS.matrix
            residual = np.linalg.norm(diff)
            return residual < tol, residual, LHS_T, RHS
