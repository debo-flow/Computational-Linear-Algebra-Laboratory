import numpy as np
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.factorizations.cholesky import CholeskyEngine

class DefinitenessClassifier:
    @staticmethod
    def is_hermitian(A: MatrixEngine, tol=1e-9):
        mat = np.array(A.matrix, dtype=complex if np.iscomplexobj(A.matrix) else float)
        return np.allclose(mat, mat.conj().T, atol=tol)

    @staticmethod
    def classify_via_eigenvalues(A: MatrixEngine, tol=1e-9):
        """
        Classifies the definiteness of a Hermitian matrix by counting eigenvalue signs.
        """
        if not A.is_square():
            raise ValueError("Definiteness classification requires a square matrix.")
            
        if not DefinitenessClassifier.is_hermitian(A, tol):
            raise ValueError("Matrix is not symmetric/Hermitian. Standard definiteness applies only to Hermitian forms.")

        mat = np.array(A.matrix, dtype=complex if np.iscomplexobj(A.matrix) else float)
        # eigh is highly stable for Hermitian matrices and guarantees real eigenvalues
        eigvals = np.linalg.eigvalsh(mat)
        
        pos_count = np.sum(eigvals > tol)
        neg_count = np.sum(eigvals < -tol)
        zero_count = np.sum(np.abs(eigvals) <= tol)
        
        n = A.rows
        
        if pos_count == n:
            status = "Positive Definite (PD)"
        elif neg_count == n:
            status = "Negative Definite (ND)"
        elif pos_count > 0 and neg_count == 0 and zero_count > 0:
            status = "Positive Semidefinite (PSD)"
        elif neg_count > 0 and pos_count == 0 and zero_count > 0:
            status = "Negative Semidefinite (NSD)"
        elif pos_count > 0 and neg_count > 0:
            status = "Indefinite"
        elif zero_count == n:
            status = "Zero Matrix (Both PSD and NSD)"
        else:
            status = "Numerically Inconclusive / Tolerance Sensitive"

        return {
            "status": status,
            "eigenvalues": eigvals,
            "lambda_min": np.min(eigvals),
            "lambda_max": np.max(eigvals),
            "pos_count": pos_count,
            "neg_count": neg_count,
            "zero_count": zero_count
        }

    @staticmethod
    def check_cholesky(A: MatrixEngine, tol=1e-9):
        """Uses Cholesky Factorization from M14 as a strict PD diagnostic."""
        try:
            CholeskyEngine.compute_cholesky(A, tol)
            return True, "Cholesky Factorization successful: Matrix is strictly Positive Definite."
        except ValueError as e:
            return False, f"Cholesky Factorization failed: {str(e)}"
