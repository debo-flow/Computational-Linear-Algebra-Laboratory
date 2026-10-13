import numpy as np
import sympy as sp
from linear_algebra.vectors.vector_engine import VectorEngine

class LinearCombinations:
    @staticmethod
    def compute(scalars, vectors):
        """Computes c1*v1 + c2*v2 + ... + ck*vk."""
        if len(scalars) != len(vectors):
            raise ValueError("The number of scalars must match the number of vectors.")
        if not vectors:
            raise ValueError("No vectors provided.")
        
        dim = vectors[0].dim
        for v in vectors:
            if v.dim != dim:
                raise ValueError("All vectors must have the same dimension.")
                
        is_symbolic = any(v.is_symbolic for v in vectors)
        
        if is_symbolic:
            result = sp.zeros(dim, 1)
            for c, v in zip(scalars, vectors):
                result += c * sp.Matrix(v.vector).reshape(dim, 1)
            return VectorEngine(list(result), True)
            
        result = np.zeros(dim)
        for c, v in zip(scalars, vectors):
            result += c * np.array(v.vector)
        return VectorEngine(result)
