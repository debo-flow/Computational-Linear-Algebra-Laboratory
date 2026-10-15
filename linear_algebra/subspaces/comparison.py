from linear_algebra.matrix.operations import MatrixOperations
from linear_algebra.vector_spaces.span_analysis import SpanAnalyzer

class SubspaceComparison:
    @staticmethod
    def compare_bases(basis_U, basis_V, tol=1e-9):
        """
        Determines if U and V span the exact same subspace.
        True iff rank(U) == rank(V) == rank([U | V]).
        """
        if not basis_U and not basis_V:
            return True
        if not basis_U or not basis_V:
            return False
            
        if basis_U[0].dim != basis_V[0].dim:
            raise ValueError("Subspaces reside in different ambient dimensions.")
            
        U_mat = SpanAnalyzer.assemble_matrix(basis_U)
        V_mat = SpanAnalyzer.assemble_matrix(basis_V)
        UV_mat = SpanAnalyzer.assemble_matrix(basis_U + basis_V)
        
        rank_U = MatrixOperations.rank(U_mat, tol)
        rank_V = MatrixOperations.rank(V_mat, tol)
        rank_UV = MatrixOperations.rank(UV_mat, tol)
        
        spans_same = (rank_U == rank_V) and (rank_U == rank_UV)
        
        return {
            "spans_same": spans_same,
            "dim_U": rank_U,
            "dim_V": rank_V,
            "dim_sum_UV": rank_UV,
            "dim_intersection": rank_U + rank_V - rank_UV
        }
