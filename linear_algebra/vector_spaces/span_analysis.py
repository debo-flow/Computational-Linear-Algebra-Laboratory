import numpy as np
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.matrix.operations import MatrixOperations
from linear_algebra.systems.system_model import LinearSystem
from linear_algebra.systems.classification import SystemClassifier

class SpanAnalyzer:
    @staticmethod
    def assemble_matrix(vectors):
        """Builds a matrix where the given vectors are column vectors."""
        if not vectors:
            raise ValueError("Cannot assemble a matrix from an empty set of vectors.")
        
        dim = vectors[0].dim
        for v in vectors:
            if v.dim != dim:
                raise ValueError("Vectors must have the same dimension to form a span.")
        
        is_sym = any(v.is_symbolic for v in vectors)
        
        if is_sym:
            import sympy as sp
            cols = [sp.Matrix(v.vector).reshape(dim, 1) for v in vectors]
            return MatrixEngine(sp.Matrix.hstack(*cols), True)
            
        mat_data = np.column_stack([v.vector for v in vectors])
        return MatrixEngine(mat_data)

    @staticmethod
    def is_in_span(target_vector, vectors, tol=1e-9):
        """Determines if target_vector is in the span of the given vectors."""
        A = SpanAnalyzer.assemble_matrix(vectors)
        # B must be a column matrix
        B_data = np.array(target_vector.vector).reshape(-1, 1)
        B = MatrixEngine(B_data, target_vector.is_symbolic)
        
        system = LinearSystem(A, B)
        report = SystemClassifier.classify(system, tol)
        
        in_span = report['solution_type'] != 'none'
        
        return in_span, report
