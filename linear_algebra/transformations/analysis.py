from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.matrix.operations import MatrixOperations
from linear_algebra.vector_spaces.subspace_analysis import SubspaceAnalyzer

class TransformationAnalyzer:
    @staticmethod
    def analyze_properties(A: MatrixEngine, tol=1e-9):
        """
        Calculates Kernel, Image, Injectivity, Surjectivity, and Bijectivity.
        Validates Rank-Nullity: dim(ker(T)) + rank(T) = dim(Domain).
        """
        rank = MatrixOperations.rank(A, tol)
        dim_domain = A.cols
        dim_codomain = A.rows
        
        nullity = dim_domain - rank
        
        # Subspace bases
        ker_basis = SubspaceAnalyzer.get_null_space(A)
        im_basis = SubspaceAnalyzer.get_column_space(A)
        
        # Classification
        is_injective = (nullity == 0)
        is_surjective = (rank == dim_codomain)
        is_bijective = is_injective and is_surjective
        
        # Geometric Scaling Effect (only for square matrices)
        scaling_factor = None
        if A.is_square():
            det = MatrixOperations.determinant(A, tol)
            if A.is_symbolic:
                import sympy as sp
                scaling_factor = sp.Abs(det)
            else:
                scaling_factor = abs(float(det))

        return {
            "rank": rank,
            "nullity": nullity,
            "dim_domain": dim_domain,
            "dim_codomain": dim_codomain,
            "is_injective": is_injective,
            "is_surjective": is_surjective,
            "is_bijective": is_bijective,
            "ker_basis": ker_basis,
            "im_basis": im_basis,
            "volume_scaling_factor": scaling_factor
        }
