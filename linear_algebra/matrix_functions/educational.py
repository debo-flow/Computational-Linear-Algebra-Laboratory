import numpy as np
import scipy.linalg as spla
from math import factorial
from linear_algebra.matrix.matrix_engine import MatrixEngine

class MatrixFunctionEducational:
    @staticmethod
    def taylor_expm(A: MatrixEngine, max_terms: int = 15):
        """
        Computes exp(A) via naive Taylor Series approximation.
        I + A + A^2/2! + ... + A^n/n!
        """
        mat = np.array(A.matrix, dtype=complex if np.iscomplexobj(A.matrix) else float)
        result = np.eye(A.rows, dtype=mat.dtype)
        current_term = np.eye(A.rows, dtype=mat.dtype)
        
        history = [{"term": 0, "residual_to_scipy": np.linalg.norm(spla.expm(mat) - result)}]
        
        for k in range(1, max_terms + 1):
            current_term = (current_term @ mat) / k
            result += current_term
            
            err = np.linalg.norm(spla.expm(mat) - result)
            history.append({"term": k, "residual_to_scipy": err})
            
        return MatrixEngine(result), history

    @staticmethod
    def eigenvalue_decomposition_function(A: MatrixEngine, func_name: str, tol=1e-9):
        """
        Applies a function via diagonalization: f(A) = V f(D) V^-1.
        Warns if matrix is defective.
        """
        mat = np.array(A.matrix, dtype=complex if np.iscomplexobj(A.matrix) else float)
        w, V = spla.eig(mat)
        
        cond_V = np.linalg.cond(V)
        if cond_V > 1 / tol:
            raise ValueError(f"Matrix is defective or poorly conditioned (Condition of V: {cond_V:.2e}). Eigenvalue decomposition fails for f(A).")
            
        func_map = {
            "exp(z)": np.exp,
            "log(z)": np.log,
            "sqrt(z)": np.sqrt,
            "sin(z)": np.sin,
            "cos(z)": np.cos
        }
        
        # Apply scalar function to eigenvalues
        f_D = np.diag(func_map[func_name](w))
        
        # Reconstruct: V f(D) V^-1
        V_inv = np.linalg.inv(V)
        f_A = V @ f_D @ V_inv
        
        # Map original to transformed eigen domain for visualization
        mapped_w = np.diag(f_D)
        
        return MatrixEngine(f_A), w, mapped_w, cond_V
