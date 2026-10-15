import numpy as np
import sympy as sp
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.matrix.operations import MatrixOperations

class EigenDiagnostics:
    @staticmethod
    def verify_trace_det(A: MatrixEngine, eigen_data, tol=1e-9):
        """
        Verifies Spectral Identities:
        Sum of eigenvalues (with AM) = Trace(A)
        Product of eigenvalues (with AM) = Det(A)
        """
        trace_A = np.trace(A.matrix) if not A.is_symbolic else sp.Matrix(A.matrix).trace()
        det_A = MatrixOperations.determinant(A, tol)
        
        sum_eig = 0
        prod_eig = 1
        
        for data in eigen_data:
            val = data['eigenvalue']
            am = data['algebraic_multiplicity']
            sum_eig += val * am
            prod_eig *= (val ** am)
            
        if A.is_symbolic:
            sum_valid = sp.simplify(sum_eig - trace_A) == 0
            prod_valid = sp.simplify(prod_eig - det_A) == 0
            return sum_valid, prod_valid, sum_eig, prod_eig, trace_A, det_A
        else:
            sum_valid = np.isclose(sum_eig, trace_A, atol=tol)
            prod_valid = np.isclose(prod_eig, det_A, atol=tol)
            return sum_valid, prod_valid, sum_eig, prod_eig, trace_A, det_A

    @staticmethod
    def check_diagonalizability(A: MatrixEngine, eigen_data):
        """
        A matrix is diagonalizable iff the sum of geometric multiplicities equals n.
        (Equivalently, algebraic multiplicity == geometric multiplicity for all eigenvalues).
        """
        n = A.rows
        total_gm = sum(data['geometric_multiplicity'] for data in eigen_data)
        is_diagonalizable = (total_gm == n)
        
        reason = (f"Diagonalizable: The sum of geometric multiplicities ({total_gm}) equals the ambient dimension n ({n}). "
                  f"There is a full basis of eigenvectors.") if is_diagonalizable else \
                 (f"Defective (Non-Diagonalizable): The sum of geometric multiplicities ({total_gm}) is strictly less than n ({n}). "
                  f"There are not enough independent eigenvectors to form a basis.")
        return is_diagonalizable, reason

    @staticmethod
    def compute_residuals(A: MatrixEngine, eigen_data):
        """
        Computes ||Av - lambda*v|| for all eigenvectors to verify correctness.
        """
        residuals = []
        for data in eigen_data:
            val = data['eigenvalue']
            for v in data['eigenvectors']:
                if A.is_symbolic:
                    Av = sp.Matrix(A.matrix) @ sp.Matrix(v.vector)
                    lv = val * sp.Matrix(v.vector)
                    res = sp.simplify(Av - lv)
                    residuals.append(0.0 if res.is_zero_matrix else float('inf'))
                else:
                    v_arr = np.array(v.vector)
                    Av = A.matrix @ v_arr
                    lv = val * v_arr
                    res = np.linalg.norm(Av - lv)
                    residuals.append(res)
        return residuals
