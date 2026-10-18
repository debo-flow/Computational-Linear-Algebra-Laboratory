import numpy as np
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.vectors.vector_engine import VectorEngine
from linear_algebra.vector_spaces.span_analysis import SpanAnalyzer
from linear_algebra.svd.core import SVDEngine
from .inner_product import InnerProductEngine

class ProjectionEngine:
    @staticmethod
    def project_onto_vector(v: VectorEngine, u: VectorEngine, tol=1e-9):
        """Projects vector v onto vector u: p = (<u, v> / <u, u>) * u"""
        norm_u_sq = np.real(InnerProductEngine.dot(u, u))
        if norm_u_sq < tol**2:
            raise ValueError("Cannot project onto a zero vector.")
            
        coeff = InnerProductEngine.dot(u, v) / norm_u_sq
        u_arr = np.array(u.vector)
        p_arr = coeff * u_arr
        
        p = VectorEngine(p_arr)
        r = VectorEngine(np.array(v.vector) - p_arr)
        return p, r, coeff

    @staticmethod
    def projection_matrix(A: MatrixEngine, tol=1e-9):
        """
        Computes orthogonal projection matrix P onto Col(A).
        Uses SVD/QR numerically stable orthonormal basis generation Q, then P = Q Q^H.
        Avoids explicit (A^H A)^-1 to bypass condition squaring.
        """
        # Get orthonormal basis using SVD
        svd_data = SVDEngine.compute(A, full_matrices=False, tol=tol)
        k = svd_data['rank']
        
        if k == 0:
            return MatrixEngine(np.zeros((A.rows, A.rows)))
            
        Q = svd_data['U'].matrix[:, :k]
        P = Q @ Q.conj().T
        
        if np.allclose(P.imag, 0): P = P.real
        return MatrixEngine(P)

    @staticmethod
    def orthogonal_decomposition(v: VectorEngine, A: MatrixEngine, tol=1e-9):
        """
        Decomposes v = p + r relative to subspace Col(A).
        Validates Pythagorean theorem: ||v||^2 ≈ ||p||^2 + ||r||^2.
        """
        P_mat = ProjectionEngine.projection_matrix(A, tol)
        v_arr = np.array(v.vector)
        p_arr = P_mat.matrix @ v_arr
        r_arr = v_arr - p_arr
        
        p = VectorEngine(p_arr)
        r = VectorEngine(r_arr)
        
        norm_v_sq = InnerProductEngine.norm(v)**2
        norm_p_sq = InnerProductEngine.norm(p)**2
        norm_r_sq = InnerProductEngine.norm(r)**2
        
        pythagorean_err = abs(norm_v_sq - (norm_p_sq + norm_r_sq))
        
        return {
            "p": p, "r": r,
            "norm_p": np.sqrt(norm_p_sq), "norm_r": np.sqrt(norm_r_sq),
            "pythagorean_valid": pythagorean_err < tol,
            "pythagorean_error": pythagorean_err
        }

    @staticmethod
    def verify_least_squares_orthogonality(A: MatrixEngine, b: VectorEngine, x: VectorEngine, tol=1e-9):
        """
        Validates Least Squares Geometry: A^H (b - Ax) ≈ 0.
        The residual r must be strictly orthogonal to the column space of A.
        """
        b_arr = np.array(b.vector)
        x_arr = np.array(x.vector)
        
        p_arr = A.matrix @ x_arr
        r_arr = b_arr - p_arr
        
        # A^H r
        A_H = A.matrix.conj().T
        normal_residual = A_H @ r_arr
        
        ortho_err = np.linalg.norm(normal_residual)
        return {
            "p": VectorEngine(p_arr),
            "r": VectorEngine(r_arr),
            "is_orthogonal": ortho_err < tol,
            "normal_residual_norm": ortho_err
        }
