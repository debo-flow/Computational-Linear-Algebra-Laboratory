from linear_algebra.matrix.operations import MatrixOperations
from .span_analysis import SpanAnalyzer

class LinearIndependenceAnalyzer:
    @staticmethod
    def analyze(vectors, tol=1e-9):
        """Determines if a set of vectors is linearly independent."""
        if not vectors:
            return {"is_independent": True, "rank": 0, "vectors_count": 0, "message": "Empty set is technically independent."}
            
        k = len(vectors)
        dim = vectors[0].dim
        
        if k > dim:
            return {
                "is_independent": False, 
                "rank": None, 
                "vectors_count": k, 
                "ambient_dimension": dim,
                "message": f"Dependent: {k} vectors in {dim}D space inherently exceed the ambient dimension."
            }
            
        A = SpanAnalyzer.assemble_matrix(vectors)
        rank = MatrixOperations.rank(A, tol)
        
        is_independent = (rank == k)
        
        msg = "Linearly Independent: Only the trivial solution (all coefficients = 0) exists." if is_independent else f"Linearly Dependent: The matrix rank ({rank}) is less than the number of vectors ({k})."
        
        return {
            "is_independent": is_independent,
            "rank": rank,
            "vectors_count": k,
            "ambient_dimension": dim,
            "message": msg
        }
