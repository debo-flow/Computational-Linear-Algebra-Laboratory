import numpy as np
import scipy.linalg as spla
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.vectors.vector_engine import VectorEngine

class MatrixDifferentialEquations:
    @staticmethod
    def solve_linear_ode(A: MatrixEngine, x0: VectorEngine, t_end: float, num_points: int = 100):
        """
        Solves dx/dt = Ax exactly using x(t) = exp(tA) x_0.
        Returns time arrays and evaluated states.
        """
        mat = np.array(A.matrix, dtype=complex if np.iscomplexobj(A.matrix) else float)
        x0_arr = np.array(x0.vector, dtype=complex if np.iscomplexobj(x0.vector) else float)
        
        t_vals = np.linspace(0, t_end, num_points)
        X_vals = np.zeros((len(t_vals), len(x0_arr)), dtype=x0_arr.dtype)
        
        for i, t in enumerate(t_vals):
            exp_tA = spla.expm(t * mat)
            X_vals[i, :] = exp_tA @ x0_arr
            
        # Determine stability from eigenvalues (max real part)
        eigvals = spla.eigvals(mat)
        max_real_part = np.max(np.real(eigvals))
        
        if max_real_part < -1e-9:
            status = "Stable (All eigenvalues have negative real parts)"
        elif max_real_part > 1e-9:
            status = "Unstable (At least one eigenvalue has a positive real part)"
        else:
            status = "Marginally Stable / Oscillatory"
            
        return t_vals, X_vals, eigvals, status
