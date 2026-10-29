import numpy as np
import scipy.linalg as spla
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.matrix.operations import MatrixOperations

class MatrixPencilAnalyzer:
    @staticmethod
    def analyze_pencil(A: MatrixEngine, B: MatrixEngine, tol=1e-9):
        """
        Analyzes the matrix pencil (A - lambda B).
        Evaluates singularity of B and performs determinant sampling.
        """
        mat_B = np.array(B.matrix, dtype=complex if np.iscomplexobj(B.matrix) else float)
        mat_A = np.array(A.matrix, dtype=complex if np.iscomplexobj(A.matrix) else float)
        
        rank_B = np.linalg.matrix_rank(mat_B, tol=tol)
        svd_B = spla.svdvals(mat_B)
        cond_B = svd_B[0] / svd_B[-1] if svd_B[-1] > tol else float('inf')
        is_B_singular = (rank_B < B.cols)
        
        # Diagnostic Determinant Sampling
        # Warning: For large matrices, det can overflow/underflow. Strictly diagnostic.
        sample_lambdas = [-1.0, 0.0, 1.0, 3.1415]
        det_samples = {}
        for lam in sample_lambdas:
            pencil_mat = MatrixEngine(mat_A - lam * mat_B)
            try:
                det_val = MatrixOperations.determinant(pencil_mat, tol)
                det_samples[lam] = det_val
            except Exception:
                det_samples[lam] = float('nan')
                
        return {
            "rank_B": rank_B,
            "singular_values_B": svd_B,
            "cond_B": cond_B,
            "is_B_singular": is_B_singular,
            "det_samples": det_samples
        }
