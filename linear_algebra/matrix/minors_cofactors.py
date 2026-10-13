import numpy as np
import sympy as sp
from .matrix_engine import MatrixEngine
from .operations import MatrixOperations

class MinorsCofactors:
    @staticmethod
    def minor_matrix(A: MatrixEngine, i: int, j: int):
        """Returns the minor matrix after deleting row i and column j."""
        if not A.is_square():
            raise ValueError("Minors are defined only for square matrices.")
        if A.rows == 1:
            # The determinant of a 0x0 matrix is mathematically 1.
            return MatrixEngine([[1]], A.is_symbolic)
            
        if A.is_symbolic:
            mat = sp.Matrix(A.matrix)
            mat.row_del(i)
            mat.col_del(j)
            return MatrixEngine(mat, True)
        
        mat = np.delete(np.delete(A.matrix, i, axis=0), j, axis=1)
        return MatrixEngine(mat)
        
    @staticmethod
    def cofactor_matrix(A: MatrixEngine):
        """Computes the matrix of cofactors."""
        if not A.is_square():
            raise ValueError("Cofactor matrix requires a square matrix.")
        
        if A.rows == 1:
            return MatrixEngine([[1]], A.is_symbolic)
            
        if A.is_symbolic:
            return MatrixEngine(sp.Matrix(A.matrix).cofactor_matrix(), True)
            
        n = A.rows
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                M_ij = MinorsCofactors.minor_matrix(A, i, j)
                det_M = MatrixOperations.determinant(M_ij)
                C[i, j] = ((-1)**(i+j)) * det_M
        return MatrixEngine(C)
        
    @staticmethod
    def adjugate(A: MatrixEngine):
        """Computes the adjugate (adjoint-transpose) matrix adj(A) = C^T."""
        if not A.is_square():
            raise ValueError("Adjugate requires a square matrix.")
        
        if A.is_symbolic:
            return MatrixEngine(sp.Matrix(A.matrix).adjugate(), True)
            
        C = MinorsCofactors.cofactor_matrix(A)
        return MatrixEngine(C.matrix.T)
