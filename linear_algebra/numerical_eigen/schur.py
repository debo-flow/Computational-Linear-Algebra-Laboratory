import numpy as np
import scipy.linalg as spla
from linear_algebra.matrix.matrix_engine import MatrixEngine

class SchurDecomposition:
    @staticmethod
    def compute(A: MatrixEngine, output='complex'):
        """
        Computes Schur Decomposition A = Z T Z^H.
        output='real' returns quasi-upper triangular T for real matrices.
        output='complex' returns strict upper triangular T.
        """
        if not A.is_square():
            raise ValueError("Schur Decomposition requires a square matrix.")
            
        mat = np.array(A.matrix, dtype=complex if np.iscomplexobj(A.matrix) else float)
        if output == 'real' and np.iscomplexobj(mat):
            raise ValueError("Real Schur form is not applicable to complex matrices.")
            
        T, Z = spla.schur(mat, output=output)
        
        Z_H = Z.conj().T
        A_recon = Z @ T @ Z_H
        
        norm_A = max(np.linalg.norm(mat, 'fro'), 1e-15)
        recon_err = np.linalg.norm(mat - A_recon, 'fro') / norm_A
        unitarity_err = np.linalg.norm(Z_H @ Z - np.eye(A.rows), 'fro')
        
        return {
            "T": MatrixEngine(T),
            "Z": MatrixEngine(Z),
            "reconstruction_error": recon_err,
            "unitarity_error": unitarity_err,
            "eigenvalues": np.diag(T) if output == 'complex' else spla.eigvals(T)
        }
