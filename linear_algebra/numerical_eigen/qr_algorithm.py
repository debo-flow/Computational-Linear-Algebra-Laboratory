import numpy as np
import scipy.linalg as spla
from linear_algebra.matrix.matrix_engine import MatrixEngine

class QREigenAlgorithm:
    @staticmethod
    def educational_qr_iteration(A: MatrixEngine, max_iter=50, tol=1e-6):
        """
        Educational unshifted QR Iteration: A_{k+1} = R_k Q_k
        Demonstrates reduction toward Schur/diagonal form.
        """
        if not A.is_square():
            raise ValueError("QR Iteration requires a square matrix.")
            
        mat = np.array(A.matrix, dtype=complex if np.iscomplexobj(A.matrix) else float)
        A_k = mat.copy()
        
        history = []
        for i in range(max_iter):
            Q, R = spla.qr(A_k)
            A_k = R @ Q
            
            # Off-diagonal norm as a convergence metric
            off_diag = A_k - np.diag(np.diag(A_k))
            # Strictly lower triangular part for Schur convergence
            lower_tri = np.tril(A_k, -1)
            off_norm = np.linalg.norm(lower_tri, 'fro')
            
            history.append({
                "iter": i+1,
                "A_k": MatrixEngine(A_k.copy()),
                "lower_off_diag_norm": off_norm
            })
            
            if off_norm < tol:
                break
                
        return MatrixEngine(A_k), history

