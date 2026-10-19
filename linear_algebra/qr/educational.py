import numpy as np
from linear_algebra.matrix.matrix_engine import MatrixEngine

class QREducational:
    @staticmethod
    def householder_qr(A: MatrixEngine):
        """Educational step-by-step Householder QR for small matrices."""
        mat = np.array(A.matrix, dtype=complex if np.iscomplexobj(A.matrix) else float)
        m, n = mat.shape
        R = mat.copy()
        Q = np.eye(m, dtype=mat.dtype)
        steps = []
        
        for k in range(min(m, n)):
            x = R[k:m, k]
            norm_x = np.linalg.norm(x)
            if norm_x < 1e-10: continue
            
            # Form Householder vector v
            if np.iscomplexobj(x):
                alpha = -np.exp(1j * np.angle(x[0])) * norm_x
            else:
                alpha = -np.sign(x[0]) * norm_x if x[0] != 0 else -norm_x
                
            e1 = np.zeros_like(x)
            e1[0] = 1.0
            u = x - alpha * e1
            v = u / np.linalg.norm(u)
            
            # Form H_k
            H_k = np.eye(m - k, dtype=mat.dtype) - 2 * np.outer(v, v.conj())
            H = np.eye(m, dtype=mat.dtype)
            H[k:, k:] = H_k
            
            R = H @ R
            Q = Q @ H.conj().T
            
            # Clean numerical noise near zero for UI rendering
            R[np.abs(R) < 1e-12] = 0
            Q[np.abs(Q) < 1e-12] = 0
            
            steps.append({
                "step": k + 1,
                "H": MatrixEngine(H.copy()),
                "Q_current": MatrixEngine(Q.copy()),
                "R_current": MatrixEngine(R.copy())
            })
            
        return MatrixEngine(Q), MatrixEngine(R), steps

    @staticmethod
    def givens_qr(A: MatrixEngine):
        """Educational step-by-step Givens Rotation QR."""
        mat = np.array(A.matrix, dtype=complex if np.iscomplexobj(A.matrix) else float)
        m, n = mat.shape
        R = mat.copy()
        Q = np.eye(m, dtype=mat.dtype)
        steps = []
        
        step_idx = 1
        for j in range(n):
            for i in range(m - 1, j, -1):
                a, b = R[i-1, j], R[i, j]
                if np.abs(b) < 1e-10: continue
                
                # Compute Givens parameters
                r = np.hypot(np.abs(a), np.abs(b))
                c = np.abs(a) / r
                if a == 0:
                    s = 1.0
                else:
                    s = (a / np.abs(a)) * np.conj(b) / r
                    
                G = np.eye(m, dtype=mat.dtype)
                G[i-1:i+1, i-1:i+1] = np.array([[c, s], [-np.conj(s), c]])
                
                R = G @ R
                Q = Q @ G.conj().T
                
                R[np.abs(R) < 1e-12] = 0
                Q[np.abs(Q) < 1e-12] = 0
                
                steps.append({
                    "step": step_idx,
                    "target": f"Zeroing R[{i},{j}]",
                    "G": MatrixEngine(G.copy()),
                    "R_current": MatrixEngine(R.copy())
                })
                step_idx += 1
                
        return MatrixEngine(Q), MatrixEngine(R), steps
