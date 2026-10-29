import numpy as np
import scipy.linalg as spla
from linear_algebra.matrix.matrix_engine import MatrixEngine

class EigenSensitivity:
    @staticmethod
    def compute_condition_numbers(A: MatrixEngine):
        """
        Computes eigenvalues and their condition numbers.
        kappa(lambda) = ||x||_2 ||y||_2 / |y^H x|
        """
        mat = np.array(A.matrix, dtype=complex if np.iscomplexobj(A.matrix) else float)
        w, vl, vr = spla.eig(mat, left=True, right=True)
        
        conditions = []
        for i in range(len(w)):
            x = vr[:, i]
            y = vl[:, i]
            
            # Normalize vectors
            x = x / np.linalg.norm(x)
            y = y / np.linalg.norm(y)
            
            dot = np.vdot(y, x) # y^H x
            if abs(dot) < 1e-15:
                cond = float('inf')
            else:
                cond = 1.0 / abs(dot)
                
            conditions.append({
                "eigenvalue": w[i],
                "condition_number": cond
            })
            
        return conditions

    @staticmethod
    def perturbation_experiment(A: MatrixEngine, noise_mag: float, seed: int = None):
        """Perturbs matrix A and tracks eigenvalue shift."""
        if seed is not None:
            np.random.seed(seed)
            
        mat = np.array(A.matrix, dtype=complex if np.iscomplexobj(A.matrix) else float)
        noise = np.random.randn(*mat.shape)
        if np.iscomplexobj(mat):
            noise = noise + 1j * np.random.randn(*mat.shape)
            
        noise = (noise / np.linalg.norm(noise, 'fro')) * noise_mag
        perturbed_mat = mat + noise
        
        orig_eigvals = spla.eigvals(mat)
        pert_eigvals = spla.eigvals(perturbed_mat)
        
        return {
            "original_eigenvalues": orig_eigvals,
            "perturbed_eigenvalues": pert_eigvals,
            "noise_matrix": MatrixEngine(noise)
        }
