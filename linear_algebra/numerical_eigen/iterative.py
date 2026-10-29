import numpy as np
import scipy.linalg as spla
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.vectors.vector_engine import VectorEngine

class IterativeEigenEngine:
    @staticmethod
    def power_iteration(A: MatrixEngine, x0: VectorEngine, max_iter=100, tol=1e-9):
        """Finds the dominant eigenvalue and eigenvector."""
        mat = np.array(A.matrix, dtype=complex if np.iscomplexobj(A.matrix) else float)
        x = np.array(x0.vector, dtype=mat.dtype)
        
        norm_x = np.linalg.norm(x)
        if norm_x < 1e-15:
            raise ValueError("Initial vector cannot be zero.")
        x = x / norm_x
        
        history = []
        lam = 0.0
        
        for i in range(max_iter):
            y = mat @ x
            y_norm = np.linalg.norm(y)
            
            if y_norm < 1e-15:
                history.append({"iter": i+1, "lambda": 0.0, "residual": 0.0})
                return 0.0, VectorEngine(x), history, True
                
            x_new = y / y_norm
            lam_new = np.vdot(x_new, mat @ x_new)
            
            residual = np.linalg.norm(mat @ x_new - lam_new * x_new)
            history.append({"iter": i+1, "lambda": lam_new, "residual": residual})
            
            if residual < tol:
                return lam_new, VectorEngine(x_new), history, True
                
            x = x_new
            lam = lam_new
            
        return lam, VectorEngine(x), history, False

    @staticmethod
    def inverse_iteration(A: MatrixEngine, x0: VectorEngine, shift: complex, max_iter=100, tol=1e-9):
        """Finds the eigenvalue closest to the shift."""
        mat = np.array(A.matrix, dtype=complex if np.iscomplexobj(A.matrix) else float)
        x = np.array(x0.vector, dtype=complex)
        
        norm_x = np.linalg.norm(x)
        if norm_x < 1e-15:
            raise ValueError("Initial vector cannot be zero.")
        x = x / norm_x
        
        I = np.eye(A.rows, dtype=complex)
        shifted_A = mat - shift * I
        
        history = []
        lam = shift
        
        for i in range(max_iter):
            try:
                y = spla.solve(shifted_A, x)
            except spla.LinAlgError:
                # Exact singularity means shift is an exact eigenvalue
                history.append({"iter": i+1, "lambda": shift, "residual": 0.0})
                return shift, VectorEngine(x), history, True
                
            y_norm = np.linalg.norm(y)
            x_new = y / y_norm
            
            lam_new = np.vdot(x_new, mat @ x_new)
            residual = np.linalg.norm(mat @ x_new - lam_new * x_new)
            history.append({"iter": i+1, "lambda": lam_new, "residual": residual})
            
            if residual < tol:
                return lam_new, VectorEngine(x_new), history, True
                
            x = x_new
            lam = lam_new
            
        return lam, VectorEngine(x), history, False

    @staticmethod
    def rayleigh_quotient_iteration(A: MatrixEngine, x0: VectorEngine, max_iter=100, tol=1e-9):
        """Finds an eigenvalue using dynamically updated shifts (Rayleigh Quotients)."""
        mat = np.array(A.matrix, dtype=complex if np.iscomplexobj(A.matrix) else float)
        x = np.array(x0.vector, dtype=complex)
        
        norm_x = np.linalg.norm(x)
        if norm_x < 1e-15:
            raise ValueError("Initial vector cannot be zero.")
        x = x / norm_x
        
        I = np.eye(A.rows, dtype=complex)
        lam = np.vdot(x, mat @ x)
        history = []
        
        for i in range(max_iter):
            shifted_A = mat - lam * I
            try:
                y = spla.solve(shifted_A, x)
            except spla.LinAlgError:
                history.append({"iter": i+1, "lambda": lam, "residual": 0.0})
                return lam, VectorEngine(x), history, True
                
            y_norm = np.linalg.norm(y)
            x_new = y / y_norm
            
            lam_new = np.vdot(x_new, mat @ x_new)
            residual = np.linalg.norm(mat @ x_new - lam_new * x_new)
            history.append({"iter": i+1, "lambda": lam_new, "residual": residual})
            
            if residual < tol:
                return lam_new, VectorEngine(x_new), history, True
                
            x = x_new
            lam = lam_new
            
        return lam, VectorEngine(x), history, False
