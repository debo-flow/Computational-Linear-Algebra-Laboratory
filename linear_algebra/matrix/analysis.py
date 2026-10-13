import numpy as np
import sympy as sp
from .matrix_engine import MatrixEngine
from .operations import MatrixOperations

class MatrixAnalysis:
    @staticmethod
    def norms(A: MatrixEngine):
        """Calculates Frobenius, 1-norm, Infinity-norm, and Spectral (2-norm)."""
        if A.is_symbolic:
            raise NotImplementedError("Norms currently supported for numerical matrices only.")
        
        return {
            "Frobenius Norm": np.linalg.norm(A.matrix, 'fro'),
            "1-Norm (Max Column Sum)": np.linalg.norm(A.matrix, 1),
            "Infinity-Norm (Max Row Sum)": np.linalg.norm(A.matrix, np.inf),
            "Spectral Norm (2-Norm)": np.linalg.norm(A.matrix, 2)
        }

    @staticmethod
    def condition_number(A: MatrixEngine, p=None):
        if A.is_symbolic:
            raise NotImplementedError("Condition number supported for numerical matrices only.")
        return np.linalg.cond(A.matrix, p)

    @staticmethod
    def structure_tests(A: MatrixEngine, tol=1e-9):
        """Tests for advanced matrix properties."""
        if not A.is_square():
            return {"Orthogonal": False, "Idempotent": False, "Involutory": False, "Nilpotent": False}
        
        results = {}
        I = MatrixEngine.generate_identity(A.rows, A.is_symbolic)
        
        # Orthogonal: A^T A = I
        A_T = MatrixEngine(A.matrix.T, A.is_symbolic)
        ATA = MatrixOperations.multiply(A_T, A)
        
        # Idempotent: A^2 = A
        A2 = MatrixOperations.multiply(A, A)
        
        if A.is_symbolic:
            results["Orthogonal"] = sp.simplify(ATA.matrix - I.matrix).is_zero_matrix
            results["Idempotent"] = sp.simplify(A2.matrix - A.matrix).is_zero_matrix
            results["Involutory"] = sp.simplify(A2.matrix - I.matrix).is_zero_matrix
            
            # Nilpotent: A^k = 0 for some k <= n
            is_nil = False
            Ak = sp.Matrix(A.matrix)
            for _ in range(A.rows):
                if Ak.is_zero_matrix:
                    is_nil = True
                    break
                Ak = Ak * sp.Matrix(A.matrix)
            results["Nilpotent"] = is_nil
        else:
            results["Orthogonal"] = np.allclose(ATA.matrix, I.matrix, atol=tol)
            results["Idempotent"] = np.allclose(A2.matrix, A.matrix, atol=tol)
            results["Involutory"] = np.allclose(A2.matrix, I.matrix, atol=tol)
            
            is_nil = False
            Ak = A.matrix
            for _ in range(A.rows):
                if np.allclose(Ak, 0, atol=tol):
                    is_nil = True
                    break
                Ak = Ak @ A.matrix
            results["Nilpotent"] = is_nil
            
        return results

    @staticmethod
    def linear_independence(A: MatrixEngine):
        """Analyzes row/col independence based on rank."""
        rank = MatrixOperations.rank(A)
        _, pivots = MatrixOperations.rref(A)
        
        return {
            "Rank": rank,
            "Full Column Rank (Independent Cols)": rank == A.cols,
            "Full Row Rank (Independent Rows)": rank == A.rows,
            "Pivot Columns": pivots
        }
