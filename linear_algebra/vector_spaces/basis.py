import sympy as sp
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.matrix.operations import MatrixOperations
from linear_algebra.vectors.vector_engine import VectorEngine
from linear_algebra.vector_spaces.span_analysis import SpanAnalyzer
from linear_algebra.vector_spaces.independence import LinearIndependenceAnalyzer

class BasisAnalyzer:
    @staticmethod
    def extract_basis(vectors):
        """
        Removes redundant vectors from a generating set to form a basis.
        Returns the basis vectors and the indices of the original vectors kept.
        """
        if not vectors:
            return [], []
            
        A = SpanAnalyzer.assemble_matrix(vectors)
        _, pivots = MatrixOperations.rref(A)
        
        basis_vectors = [vectors[i] for i in pivots]
        return basis_vectors, pivots

    @staticmethod
    def validate_ambient_basis(vectors, target_dim, tol=1e-9):
        """
        Validates if a set of vectors forms a valid basis for R^n (target_dim).
        """
        if len(vectors) != target_dim:
            return False, f"Dimension mismatch: Expected {target_dim} vectors, got {len(vectors)}."
            
        report = LinearIndependenceAnalyzer.analyze(vectors, tol)
        if not report["is_independent"]:
            return False, "Vectors are linearly dependent, so they do not span the space."
            
        return True, f"Valid Basis: {target_dim} linearly independent vectors span R^{target_dim}."

    @staticmethod
    def get_row_space_basis(A: MatrixEngine):
        """Extracts the row space basis using the non-zero rows of the RREF."""
        rref_mat, _ = MatrixOperations.rref(A)
        basis_rows = []
        sym_mat = sp.Matrix(rref_mat.matrix)
        
        for i in range(sym_mat.rows):
            if not sym_mat.row(i).is_zero_matrix:
                basis_rows.append(VectorEngine(list(sym_mat.row(i)), True))
                
        return basis_rows
