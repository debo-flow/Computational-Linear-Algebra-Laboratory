import numpy as np
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.matrix.operations import MatrixOperations

class SylvesterCriterion:
    @staticmethod
    def analyze(A: MatrixEngine, tol=1e-9):
        """
        Computes all Leading Principal Minors (LPMs) to apply Sylvester's Criterion.
        - PD: All LPMs > 0
        - ND: LPMs alternate signs (-1)^k det(A_k) > 0
        """
        mat = np.array(A.matrix, dtype=complex if np.iscomplexobj(A.matrix) else float)
        n = mat.shape[0]
        
        minors = []
        is_pd = True
        is_nd = True
        
        for k in range(1, n + 1):
            sub_mat = MatrixEngine(mat[:k, :k])
            det_val = MatrixOperations.determinant(sub_mat, tol)
            
            # Since matrix should be Hermitian, determinants are strictly real mathematically
            real_det = np.real(det_val) if np.iscomplexobj(det_val) else float(det_val)
            minors.append(real_det)
            
            # PD Check
            if real_det <= tol:
                is_pd = False
                
            # ND Check (Alternating: - + - +)
            expected_sign = -1 if k % 2 != 0 else 1
            if (expected_sign * real_det) <= tol:
                is_nd = False
                
        status = "Indefinite or Semidefinite (Sylvester checks strict PD/ND only)"
        if is_pd:
            status = "Strictly Positive Definite"
        elif is_nd:
            status = "Strictly Negative Definite"
            
        return {
            "leading_principal_minors": minors,
            "is_pd": is_pd,
            "is_nd": is_nd,
            "status": status
        }
