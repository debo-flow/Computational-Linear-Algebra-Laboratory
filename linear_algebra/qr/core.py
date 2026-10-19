import numpy as np
import scipy.linalg as spla
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.orthogonality.gram_schmidt import GramSchmidtEngine
from linear_algebra.vectors.vector_engine import VectorEngine

class QREngine:
    @staticmethod
    def compute_library_qr(A: MatrixEngine, mode='reduced', tol=1e-9):
        """
        Computes QR factorization using LAPACK/SciPy.
        Mode can be 'reduced' (economic) or 'full'.
        """
        mat = np.array(A.matrix, dtype=complex if np.iscomplexobj(A.matrix) else float)
        scipy_mode = 'economic' if mode == 'reduced' else 'full'
        
        Q, R = spla.qr(mat, mode=scipy_mode)
        
        # Verify A ≈ QR
        A_recon = Q @ R
        norm_A = max(np.linalg.norm(mat, 'fro'), 1e-15)
        recon_err = np.linalg.norm(mat - A_recon, 'fro') / norm_A
        
        # Verify Q^H Q ≈ I
        I_approx = Q.conj().T @ Q
        ortho_err = np.linalg.norm(I_approx - np.eye(Q.shape[1]), 'fro')
        
        return {
            "Q": MatrixEngine(Q),
            "R": MatrixEngine(R),
            "reconstruction_error": recon_err,
            "orthogonality_error": ortho_err
        }

    @staticmethod
    def compute_pivoted_qr(A: MatrixEngine, tol=1e-9):
        """
        Computes Rank-Revealing Column-Pivoted QR: A[:, P] = Q R
        """
        mat = np.array(A.matrix, dtype=complex if np.iscomplexobj(A.matrix) else float)
        Q, R, P = spla.qr(mat, pivoting=True, mode='economic')
        
        # Numerical rank via diagonal of R
        diag_R = np.abs(np.diag(R))
        num_rank = np.sum(diag_R > tol)
        
        return {
            "Q": MatrixEngine(Q),
            "R": MatrixEngine(R),
            "P": P,  # 1D array of permutation indices
            "rank": num_rank,
            "diag_R": diag_R
        }

    @staticmethod
    def compute_gram_schmidt_qr(A: MatrixEngine, method='modified', tol=1e-9):
        """
        Computes Reduced QR via Classical or Modified Gram-Schmidt.
        Builds R iteratively via inner products R = Q^H A.
        """
        mat = np.array(A.matrix, dtype=complex if np.iscomplexobj(A.matrix) else float)
        cols = [VectorEngine(list(mat[:, i])) for i in range(A.cols)]
        
        gs_report = GramSchmidtEngine.compute(cols, method=method, tol=tol)
        
        if gs_report['skipped_indices']:
            raise ValueError(f"Matrix is rank deficient. {method.capitalize()} Gram-Schmidt requires full column rank.")
            
        Q_cols = gs_report['orthonormal_vectors']
        Q_mat = np.column_stack([np.array(q.vector) for q in Q_cols])
        
        # R = Q^H A (upper triangular mathematically guaranteed by GS)
        R_mat = Q_mat.conj().T @ mat
        # Clean lower triangular numerical noise
        R_mat = np.triu(R_mat)
        
        recon_err = np.linalg.norm(mat - (Q_mat @ R_mat), 'fro') / max(np.linalg.norm(mat, 'fro'), 1e-15)
        
        return {
            "Q": MatrixEngine(Q_mat),
            "R": MatrixEngine(R_mat),
            "orthogonality_error": gs_report['orthogonality_error'],
            "reconstruction_error": recon_err
        }
