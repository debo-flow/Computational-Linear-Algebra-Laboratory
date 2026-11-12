import numpy as np
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.matrix.operations import MatrixOperations

class MatrixCalculusIdentities:
    @staticmethod
    def evaluate_identities(X: MatrixEngine, A: MatrixEngine=None):
        """
        Evaluates exact Matrix Calculus identities evaluated at Matrix X.
        """
        X_mat = np.array(X.matrix, dtype=float)
        results = {}
        
        # 1. d(Tr(X)) / dX = I
        results['d_trX'] = np.eye(X.rows)
        
        # 2. d(Det(X)) / dX = det(X) * X^{-T} (Requires invertible X)
        try:
            det_X = np.linalg.det(X_mat)
            inv_X_T = np.linalg.inv(X_mat).T
            results['d_detX'] = det_X * inv_X_T
            
            # 3. d(log(Det(X))) / dX = X^{-T}
            if det_X > 0:
                results['d_logdetX'] = inv_X_T
            else:
                results['d_logdetX'] = "Undefined (Det(X) <= 0)"
                
            # 4. d(Tr(X^{-1})) / dX = -(X^{-2})^T
            X_inv2 = np.linalg.matrix_power(np.linalg.inv(X_mat), 2)
            results['d_trXinv'] = -X_inv2.T
        except np.linalg.LinAlgError:
            results['d_detX'] = "Undefined (Singular X)"
            results['d_logdetX'] = "Undefined (Singular X)"
            results['d_trXinv'] = "Undefined (Singular X)"
            
        if A is not None and A.shape == X.shape:
            A_mat = np.array(A.matrix, dtype=float)
            # 5. d(Tr(AX)) / dX = A^T
            results['d_trAX'] = A_mat.T
            
            # 6. d(Tr(X^T A X)) / dX = (A + A^T) X
            results['d_trXtAX'] = (A_mat + A_mat.T) @ X_mat
            
        return results
