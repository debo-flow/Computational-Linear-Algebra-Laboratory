import numpy as np
import scipy.linalg as spla
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.vectors.vector_engine import VectorEngine

class GeneralizedEigenEngine:
    @staticmethod
    def compute_general(A: MatrixEngine, B: MatrixEngine, tol=1e-9):
        """
        Solves Av = lambda Bv for general matrix pairs (A, B).
        Uses homogeneous coordinates (alpha, beta) to safely represent infinite eigenvalues.
        """
        if not A.is_square() or not B.is_square() or A.shape != B.shape:
            raise ValueError("A and B must be square matrices of the exact same dimensions.")
            
        mat_A = np.array(A.matrix, dtype=complex if np.iscomplexobj(A.matrix) else float)
        mat_B = np.array(B.matrix, dtype=complex if np.iscomplexobj(B.matrix) else float)
        
        try:
            # Use homogeneous_eigvals=True to return (alpha, beta) representation safely
            # w is shape (2, M): w[0] = alpha, w[1] = beta. lambda = alpha / beta.
            w, vr = spla.eig(mat_A, mat_B, homogeneous_eigvals=True)
            alpha = w[0]
            beta = w[1]
        except TypeError:
            # Fallback for older SciPy versions that lack homogeneous_eigvals
            eigvals, vr = spla.eig(mat_A, mat_B)
            alpha = eigvals
            beta = np.ones_like(eigvals)
            # Find infs/NaNs and construct pseudo-homogeneous coords
            for i, val in enumerate(eigvals):
                if not np.isfinite(val):
                    alpha[i] = 1.0
                    beta[i] = 0.0

        results = []
        for i in range(len(alpha)):
            a_val, b_val = alpha[i], beta[i]
            v = vr[:, i]
            
            # Classification
            if abs(b_val) > tol:
                eigval = a_val / b_val
                status = "Finite"
                # Residual: ||Av - lambda Bv||
                res_vec = mat_A @ v - eigval * (mat_B @ v)
                residual = np.linalg.norm(res_vec)
            elif abs(a_val) > tol:
                eigval = complex(float('inf'), 0)
                status = "Infinite (Singular B)"
                # Residual for infinite: ||beta Av - alpha Bv|| = ||0 - alpha Bv|| = ||alpha Bv|| ~ 0
                res_vec = b_val * (mat_A @ v) - a_val * (mat_B @ v)
                residual = np.linalg.norm(res_vec)
            else:
                eigval = complex(float('nan'), float('nan'))
                status = "Indeterminate (0/0)"
                res_vec = b_val * (mat_A @ v) - a_val * (mat_B @ v)
                residual = np.linalg.norm(res_vec)
                
            results.append({
                "alpha": a_val,
                "beta": b_val,
                "eigenvalue": eigval,
                "status": status,
                "eigenvector": VectorEngine(list(v)),
                "residual": residual
            })
            
        return results

    @staticmethod
    def compute_symmetric_definite(A: MatrixEngine, B: MatrixEngine, tol=1e-9):
        """
        Solves Av = lambda Bv for Symmetric/Hermitian A and Symmetric-Definite B.
        Uses scipy.linalg.eigh which leverages fast, robust Cholesky decomposition of B natively.
        """
        if not A.is_square() or not B.is_square() or A.shape != B.shape:
            raise ValueError("A and B must be square matrices of the exact same dimensions.")
            
        mat_A = np.array(A.matrix, dtype=complex if np.iscomplexobj(A.matrix) else float)
        mat_B = np.array(B.matrix, dtype=complex if np.iscomplexobj(B.matrix) else float)
        
        # Validation
        if not np.allclose(mat_A, mat_A.conj().T, atol=tol):
            raise ValueError("Matrix A must be Symmetric/Hermitian.")
        if not np.allclose(mat_B, mat_B.conj().T, atol=tol):
            raise ValueError("Matrix B must be Symmetric/Hermitian.")
            
        try:
            # eigh handles the Cholesky factorization of B internally
            eigvals, vr = spla.eigh(mat_A, mat_B)
        except spla.LinAlgError:
            raise ValueError("Matrix B is not Positive-Definite. The Symmetric-Definite solver requires B > 0.")
            
        # Verify B-Orthogonality: V^H B V ≈ I
        V_H_B_V = vr.conj().T @ mat_B @ vr
        b_ortho_error = np.linalg.norm(V_H_B_V - np.eye(A.rows), 'fro')
        
        results = []
        for i, val in enumerate(eigvals):
            v = vr[:, i]
            res_vec = mat_A @ v - val * (mat_B @ v)
            residual = np.linalg.norm(res_vec)
            
            results.append({
                "eigenvalue": val,
                "eigenvector": VectorEngine(list(v)),
                "residual": residual
            })
            
        return results, b_ortho_error
