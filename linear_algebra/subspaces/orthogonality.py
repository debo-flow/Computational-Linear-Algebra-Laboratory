import numpy as np
import sympy as sp
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.vectors.vector_engine import VectorEngine

class SubspaceOrthogonality:
    @staticmethod
    def verify_orthogonality(basis_U, basis_V, tol=1e-9):
        """
        Verifies if Subspace U is orthogonal to Subspace V.
        Calculates Hermitian inner products.
        """
        if not basis_U or not basis_V:
            return True, 0.0  # Trivially orthogonal if one is empty
            
        max_residual = 0.0
        is_symbolic = basis_U[0].is_symbolic or basis_V[0].is_symbolic
        
        for u in basis_U:
            for v in basis_V:
                if is_symbolic:
                    # u^H * v
                    dot_val = sp.simplify(sp.Matrix(u.vector).H @ sp.Matrix(v.vector))[0, 0]
                    if dot_val != 0:
                        return False, float(sp.Abs(dot_val))
                else:
                    u_arr = np.array(u.vector)
                    v_arr = np.array(v.vector)
                    # Hermitian dot product: dot(conj(u), v)
                    dot_val = np.vdot(u_arr, v_arr)
                    if abs(dot_val) > max_residual:
                        max_residual = abs(dot_val)
                        
        return max_residual < tol, max_residual
