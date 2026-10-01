import numpy as np
import sympy as sp
from .matrix_engine import MatrixEngine

class MatrixOperations:
    @staticmethod
    def add(A: MatrixEngine, B: MatrixEngine):
        if A.shape != B.shape:
            raise ValueError(f"Addition undefined: Dimensions {A.shape} and {B.shape} must match exactly.")
        if A.is_symbolic or B.is_symbolic:
            return MatrixEngine(sp.Matrix(A.matrix) + sp.Matrix(B.matrix), True)
        return MatrixEngine(A.matrix + B.matrix)

    @staticmethod
    def multiply(A: MatrixEngine, B: MatrixEngine):
        if A.cols != B.rows:
            raise ValueError(f"Multiplication undefined: A columns ({A.cols}) must equal B rows ({B.rows}).")
        if A.is_symbolic or B.is_symbolic:
            return MatrixEngine(sp.Matrix(A.matrix) * sp.Matrix(B.matrix), True)
        return MatrixEngine(A.matrix @ B.matrix)

    @staticmethod
    def determinant(A: MatrixEngine, tol=1e-9):
        if not A.is_square():
            raise ValueError("Determinant undefined: Matrix must be square (n × n).")
        if A.is_symbolic:
            return A.matrix.det()
        return np.linalg.det(A.matrix)

    @staticmethod
    def inverse(A: MatrixEngine, tol=1e-9):
        if not A.is_square():
            raise ValueError("Inverse undefined: Matrix must be square.")
        
        det_val = MatrixOperations.determinant(A)
        if abs(float(det_val)) < tol:
            raise ValueError("Inverse undefined: det(A) = 0 implies the matrix is singular.")
            
        if A.is_symbolic:
            return MatrixEngine(A.matrix.inv(), True)
        return MatrixEngine(np.linalg.inv(A.matrix))

    @staticmethod
    def rank(A: MatrixEngine, tol=1e-9):
        if A.is_symbolic:
            return A.matrix.rank()
        return np.linalg.matrix_rank(A.matrix, tol=tol)

    @staticmethod
    def rref(A: MatrixEngine):
        """Returns RREF matrix and pivot columns."""
        sym_mat = sp.Matrix(A.matrix)
        rref_mat, pivots = sym_mat.rref()
        return MatrixEngine(rref_mat, True), pivots

    @staticmethod
    def solve_system(A: MatrixEngine, b: MatrixEngine):
        """Solves Ax = b and provides consistency analysis."""
        if A.rows != b.rows or b.cols != 1:
            raise ValueError("Invalid system: b must be a column vector matching A's row count.")
        
        # Augmented matrix [A | b]
        aug_data = np.hstack((A.matrix, b.matrix)) if not A.is_symbolic else A.matrix.row_join(b.matrix)
        Aug = MatrixEngine(aug_data, A.is_symbolic)
        
        rank_A = MatrixOperations.rank(A)
        rank_Aug = MatrixOperations.rank(Aug)
        
        consistency_report = {
            "rank_A": rank_A,
            "rank_Aug": rank_Aug,
            "unknowns": A.cols,
            "status": ""
        }
        
        if rank_A < rank_Aug:
            consistency_report["status"] = "No Solution (Inconsistent: rank(A) < rank([A|b]))"
            return None, consistency_report
        elif rank_A == A.cols:
            consistency_report["status"] = "Unique Solution (Consistent: rank(A) = rank([A|b]) = n)"
            if A.is_symbolic:
                x = A.matrix.LUsolve(b.matrix)
                return MatrixEngine(x, True), consistency_report
            else:
                x = np.linalg.solve(A.matrix, b.matrix)
                return MatrixEngine(x), consistency_report
        else:
            consistency_report["status"] = "Infinitely Many Solutions (Consistent: rank(A) = rank([A|b]) < n)"
            # RREF approach for parametric solutions
            rref_aug, _ = MatrixOperations.rref(Aug)
            return rref_aug, consistency_report
