import numpy as np
from linear_algebra.vectors.vector_engine import VectorEngine
from linear_algebra.matrix.matrix_engine import MatrixEngine

class InnerProductEngine:
    @staticmethod
    def dot(u: VectorEngine, v: VectorEngine):
        """
        Computes the inner product <u, v>. 
        For complex vectors, enforces the Hermitian convention: <u, v> = u^H v.
        """
        if u.dim != v.dim:
            raise ValueError(f"Vector dimension mismatch: {u.dim} and {v.dim}.")
        u_arr = np.array(u.vector, dtype=complex if np.iscomplexobj(u.vector) else float)
        v_arr = np.array(v.vector, dtype=complex if np.iscomplexobj(v.vector) else float)
        # np.vdot naturally conjugates the first argument: u^H @ v
        return np.vdot(u_arr, v_arr)

    @staticmethod
    def norm(u: VectorEngine):
        """Computes the L2 norm (Euclidean/Hermitian length) of a vector."""
        return np.sqrt(np.real(InnerProductEngine.dot(u, u)))

    @staticmethod
    def normalize(u: VectorEngine, tol=1e-9):
        """Normalizes a vector. Safely handles zero vectors by raising an error."""
        n = InnerProductEngine.norm(u)
        if n < tol:
            raise ValueError("Cannot normalize a zero or near-zero vector.")
        u_arr = np.array(u.vector)
        return VectorEngine(u_arr / n)

    @staticmethod
    def check_orthogonality(u: VectorEngine, v: VectorEngine, tol=1e-9):
        """Checks if two vectors are orthogonal (<u, v> ≈ 0)."""
        val = InnerProductEngine.dot(u, v)
        return abs(val) < tol, val

    @staticmethod
    def gram_matrix(vectors):
        """Computes the Gram matrix G where G_ij = <v_i, v_j>."""
        k = len(vectors)
        if k == 0:
            return MatrixEngine([])
        G = np.zeros((k, k), dtype=complex if any(np.iscomplexobj(v.vector) for v in vectors) else float)
        for i in range(k):
            for j in range(k):
                G[i, j] = InnerProductEngine.dot(vectors[i], vectors[j])
        
        # Strip imaginary parts if strictly real
        if np.allclose(G.imag, 0):
            G = G.real
        return MatrixEngine(G)

    @staticmethod
    def analyze_orthonormal_set(vectors, tol=1e-9):
        """Determines if a vector set is orthogonal or orthonormal."""
        G_engine = InnerProductEngine.gram_matrix(vectors)
        G = G_engine.matrix
        k = len(vectors)
        
        I = np.eye(k)
        
        is_orthogonal = True
        is_orthonormal = True
        max_error = 0.0
        
        for i in range(k):
            for j in range(k):
                expected = 1.0 if i == j else 0.0
                err = abs(G[i, j] - expected)
                if err > max_error: max_error = err
                
                if i != j and abs(G[i, j]) > tol:
                    is_orthogonal = False
                    is_orthonormal = False
                if i == j and abs(G[i, j] - 1.0) > tol:
                    is_orthonormal = False
                    
        return {
            "is_orthogonal": is_orthogonal,
            "is_orthonormal": is_orthonormal,
            "max_orthogonality_error": max_error,
            "gram_matrix": G_engine
        }
