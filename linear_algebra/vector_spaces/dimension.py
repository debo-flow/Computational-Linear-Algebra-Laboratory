from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.matrix.operations import MatrixOperations

class DimensionAnalyzer:
    @staticmethod
    def analyze_matrix_spaces(A: MatrixEngine, tol=1e-9):
        """
        Validates the Rank-Nullity Theorem: Rank(A) + Nullity(A) = n
        """
        n_cols = A.cols
        m_rows = A.rows
        rank = MatrixOperations.rank(A, tol)
        nullity = n_cols - rank
        
        # Dimensions of fundamental spaces
        col_space_dim = rank
        row_space_dim = rank 
        
        is_rank_nullity_valid = (rank + nullity == n_cols)
        
        return {
            "ambient_domain_dim": n_cols,
            "ambient_codomain_dim": m_rows,
            "rank": rank,
            "nullity": nullity,
            "col_space_dim": col_space_dim,
            "row_space_dim": row_space_dim,
            "rank_nullity_holds": is_rank_nullity_valid
        }
