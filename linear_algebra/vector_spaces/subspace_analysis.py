import sympy as sp
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.vectors.vector_engine import VectorEngine

class SubspaceAnalyzer:
    @staticmethod
    def get_column_space(A: MatrixEngine):
        """Extracts basis vectors for the column space (Image)."""
        sym_A = sp.Matrix(A.matrix)
        col_basis = sym_A.columnspace()
        return [VectorEngine(list(c), True) for c in col_basis]

    @staticmethod
    def get_null_space(A: MatrixEngine):
        """Extracts basis vectors for the null space (Kernel) representing Ax = 0."""
        sym_A = sp.Matrix(A.matrix)
        null_basis = sym_A.nullspace()
        return [VectorEngine(list(n), True) for n in null_basis]
