import numpy as np
import scipy.linalg as spla
from linear_algebra.matrix.matrix_engine import MatrixEngine

class CholeskyEngine:
    @staticmethod
    def compute_cholesky(A: MatrixEngine, tol=1e-9):
        """
        Computes Cholesky Factorization A = L L^H.
        Requires Symmetric/Hermitian Positive-Definite matrix.
        """
        mat = np.array(A.matrix, dtype=complex if np.iscomplexobj(A.matrix) else float)
        if not A.is_square():
            raise ValueError("Cholesky Factorization requires a square matrix.")
            
        if not np.allclose(mat, mat.conj().T, atol=tol):
            raise ValueError("Matrix is not Symmetric (or Hermitian). Cholesky Factorization is mathematically undefined.")
            
        try:
            L = spla.cholesky(mat, lower=True)
            A_recon = L @ L.conj().T
            recon_err = np.linalg.norm(mat - A_recon, 'fro') / max(np.linalg.norm(mat, 'fro'), 1e-15)
            
            return {
                "L": MatrixEngine(L),
                "L_H": MatrixEngine(L.conj().T),
                "reconstruction_error": recon_err
            }
        except spla.LinAlgError:
            raise ValueError("Matrix is not Positive-Definite. Leading minor is not strictly positive.")

    @staticmethod
    def compute_ldl(A: MatrixEngine, tol=1e-9):
        """
        Computes LDL^H Factorization A = P L D L^H P^T.
        Robust to Symmetric Indefinite matrices.
        """
        mat = np.array(A.matrix, dtype=complex if np.iscomplexobj(A.matrix) else float)
        if not np.allclose(mat, mat.conj().T, atol=tol):
            raise ValueError("Matrix is not Symmetric/Hermitian. LDL Factorization requires symmetry.")
            
        # SciPy ldl returns L, D, perm such that A[perm, :][:, perm] = L @ D @ L^H
        # This implies P_mat.T @ A @ P_mat = L @ D @ L^H, hence A = P_mat @ L @ D @ L^H @ P_mat.T
        L, D, perm = spla.ldl(mat, lower=True)
        
        n = mat.shape[0]
        P_mat = np.eye(n)[:, perm] # Permutation matrix formulation
        P_out = P_mat
        
        A_recon = P_out @ L @ D @ L.conj().T @ P_out.T
        recon_err = np.linalg.norm(mat - A_recon, 'fro') / max(np.linalg.norm(mat, 'fro'), 1e-15)
        
        return {
            "P": MatrixEngine(P_out),
            "L": MatrixEngine(L),
            "D": MatrixEngine(D),
            "reconstruction_error": recon_err
        }
