import numpy as np

class KrylovSubspace:
    @staticmethod
    def arnoldi_iteration(A, v, k: int):
        """
        Computes the Arnoldi iteration for a general matrix A.
        Returns orthogonal basis Q and upper Hessenberg matrix H.
        """
        n = A.shape[0]
        Q = np.zeros((n, k+1), dtype=complex if np.iscomplexobj(A) or np.iscomplexobj(v) else float)
        H = np.zeros((k+1, k), dtype=Q.dtype)
        
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-15:
            raise ValueError("Initial vector cannot be zero.")
            
        Q[:, 0] = v / v_norm
        
        for j in range(k):
            w = A @ Q[:, j]
            for i in range(j + 1):
                H[i, j] = np.vdot(Q[:, i], w)
                w = w - H[i, j] * Q[:, i]
                
            H[j+1, j] = np.linalg.norm(w)
            if H[j+1, j] < 1e-12:
                # Breakdown - invariant subspace found
                return Q[:, :j+1], H[:j+1, :j]
                
            Q[:, j+1] = w / H[j+1, j]
            
        return Q, H

    @staticmethod
    def lanczos_iteration(A, v, k: int):
        """
        Computes the Lanczos iteration for a symmetric/Hermitian matrix A.
        Returns orthogonal basis Q and tridiagonal matrix T.
        """
        n = A.shape[0]
        Q = np.zeros((n, k+1), dtype=float)
        alpha = np.zeros(k, dtype=float)
        beta = np.zeros(k, dtype=float)
        
        v_norm = np.linalg.norm(v)
        Q[:, 0] = v / v_norm
        w = A @ Q[:, 0]
        alpha[0] = np.vdot(Q[:, 0], w)
        w = w - alpha[0] * Q[:, 0]
        
        for j in range(1, k):
            beta[j-1] = np.linalg.norm(w)
            if beta[j-1] < 1e-12:
                break
                
            Q[:, j] = w / beta[j-1]
            w = A @ Q[:, j]
            alpha[j] = np.vdot(Q[:, j], w)
            w = w - alpha[j] * Q[:, j] - beta[j-1] * Q[:, j-1]
            
        # Construct Tridiagonal T
        T = np.diag(alpha[:k]) + np.diag(beta[:k-1], 1) + np.diag(beta[:k-1], -1)
        return Q[:, :k], T
