import numpy as np
import scipy.linalg as spla
from linear_algebra.matrix.matrix_engine import MatrixEngine

class MatrixFunctionEngine:
    @staticmethod
    def compute_expm(A: MatrixEngine, tol=1e-12):
        """Computes exp(A) using Padé approximation / scaling and squaring."""
        if not A.is_square():
            raise ValueError("Matrix exponential requires a square matrix.")
        
        mat = np.array(A.matrix, dtype=complex if np.iscomplexobj(A.matrix) else float)
        exp_A = spla.expm(mat)
        
        # Clean numeric noise if strictly real
        if not np.iscomplexobj(mat) and np.allclose(exp_A.imag, 0, atol=tol):
            exp_A = exp_A.real
            
        return MatrixEngine(exp_A)

    @staticmethod
    def compute_logm(A: MatrixEngine, tol=1e-12):
        """
        Computes the principal matrix logarithm log(A).
        Checks reconstruction: ||exp(log(A)) - A||
        """
        if not A.is_square():
            raise ValueError("Matrix logarithm requires a square matrix.")
            
        mat = np.array(A.matrix, dtype=complex if np.iscomplexobj(A.matrix) else float)
        
        # Check singularity which mathematically breaks logm
        if abs(spla.det(mat)) < tol:
            raise ValueError("Matrix is computationally singular. Logarithm does not exist.")

        log_A = spla.logm(mat)
        
        # Validation
        exp_log_A = spla.expm(log_A)
        norm_A = max(np.linalg.norm(mat, 'fro'), 1e-15)
        reconstruction_error = np.linalg.norm(exp_log_A - mat, 'fro') / norm_A
        
        return MatrixEngine(log_A), reconstruction_error

    @staticmethod
    def compute_sqrtm(A: MatrixEngine, tol=1e-12):
        """
        Computes the principal matrix square root sqrt(A).
        Checks reconstruction: ||sqrt(A)^2 - A||
        """
        if not A.is_square():
            raise ValueError("Matrix square root requires a square matrix.")
            
        mat = np.array(A.matrix, dtype=complex if np.iscomplexobj(A.matrix) else float)
        sqrt_A = spla.sqrtm(mat)
        
        # Validation
        recon = sqrt_A @ sqrt_A
        norm_A = max(np.linalg.norm(mat, 'fro'), 1e-15)
        reconstruction_error = np.linalg.norm(recon - mat, 'fro') / norm_A
        
        # Real input can have complex square roots (e.g. negative diagonals)
        if not np.iscomplexobj(mat) and np.allclose(sqrt_A.imag, 0, atol=tol):
            sqrt_A = sqrt_A.real
            
        return MatrixEngine(sqrt_A), reconstruction_error

    @staticmethod
    def compute_fractional_power(A: MatrixEngine, alpha: float, tol=1e-12):
        """Computes A^alpha."""
        if not A.is_square():
            raise ValueError("Fractional matrix power requires a square matrix.")
            
        mat = np.array(A.matrix, dtype=complex if np.iscomplexobj(A.matrix) else float)
        pow_A = spla.fractional_matrix_power(mat, alpha)
        
        if not np.iscomplexobj(mat) and np.allclose(pow_A.imag, 0, atol=tol):
            pow_A = pow_A.real
            
        return MatrixEngine(pow_A)

    @staticmethod
    def compute_general(A: MatrixEngine, func_name: str, tol=1e-12):
        """Computes specific trig/hyperbolic matrix functions."""
        if not A.is_square():
            raise ValueError(f"{func_name} requires a square matrix.")
            
        mat = np.array(A.matrix, dtype=complex if np.iscomplexobj(A.matrix) else float)
        
        func_map = {
            "sin(A)": spla.sinm,
            "cos(A)": spla.cosm,
            "tan(A)": spla.tanm,
            "sinh(A)": spla.sinhm,
            "cosh(A)": spla.coshm,
            "tanh(A)": spla.tanhm
        }
        
        if func_name not in func_map:
            raise ValueError(f"Function {func_name} is not supported.")
            
        res = func_map[func_name](mat)
        
        if not np.iscomplexobj(mat) and np.allclose(res.imag, 0, atol=tol):
            res = res.real
            
        return MatrixEngine(res)
