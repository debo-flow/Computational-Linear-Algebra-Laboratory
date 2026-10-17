import numpy as np
from linear_algebra.matrix.matrix_engine import MatrixEngine
from .core import SVDEngine

class PseudoinverseEngine:
    @staticmethod
    def compute(A: MatrixEngine, tol=1e-9):
        """
        Computes the Moore-Penrose pseudoinverse A^+ = V Σ^+ U^H
        """
        mat = np.array(A.matrix, dtype=float) if not np.iscomplexobj(A.matrix) else np.array(A.matrix, dtype=complex)
        
        # Utilize numpy's robust pinv which is intrinsically SVD-based
        A_plus = np.linalg.pinv(mat, rcond=tol)
        return MatrixEngine(A_plus)

    @staticmethod
    def verify_mp_conditions(A: MatrixEngine, A_plus: MatrixEngine, tol=1e-9):
        """
        Verifies the four Moore-Penrose conditions.
        1. A A^+ A = A
        2. A^+ A A^+ = A^+
        3. (A A^+)^H = A A^+
        4. (A^+ A)^H = A^+ A
        """
        M = np.array(A.matrix)
        Mp = np.array(A_plus.matrix)
        
        # 1. A A^+ A ≈ A
        cond1_mat = M @ Mp @ M
        cond1_err = np.linalg.norm(cond1_mat - M)
        
        # 2. A^+ A A^+ ≈ A^+
        cond2_mat = Mp @ M @ Mp
        cond2_err = np.linalg.norm(cond2_mat - Mp)
        
        # 3. (A A^+)^H ≈ A A^+ (Hermitian)
        AAp = M @ Mp
        cond3_err = np.linalg.norm(AAp.conj().T - AAp)
        
        # 4. (A^+ A)^H ≈ A^+ A (Hermitian)
        ApA = Mp @ M
        cond4_err = np.linalg.norm(ApA.conj().T - ApA)
        
        return {
            "cond1_valid": cond1_err < tol, "cond1_err": cond1_err,
            "cond2_valid": cond2_err < tol, "cond2_err": cond2_err,
            "cond3_valid": cond3_err < tol, "cond3_err": cond3_err,
            "cond4_valid": cond4_err < tol, "cond4_err": cond4_err,
            "all_valid": (cond1_err < tol and cond2_err < tol and cond3_err < tol and cond4_err < tol)
        }

    @staticmethod
    def svd_least_squares(A: MatrixEngine, b: MatrixEngine, tol=1e-9):
        """
        Computes the minimum-norm least-squares solution x = A^+ b.
        """
        A_plus = PseudoinverseEngine.compute(A, tol)
        x_num = np.array(A_plus.matrix) @ np.array(b.matrix)
        x = MatrixEngine(x_num)
        
        # Residuals r = b - Ax
        b_num = np.array(b.matrix)
        Ax = np.array(A.matrix) @ x_num
        residual_vec = b_num - Ax
        residual_norm = np.linalg.norm(residual_vec)
        solution_norm = np.linalg.norm(x_num)
        
        return x, residual_norm, solution_norm
