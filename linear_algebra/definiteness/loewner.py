import numpy as np
from linear_algebra.matrix.matrix_engine import MatrixEngine
from .classifier import DefinitenessClassifier

class LoewnerOrder:
    @staticmethod
    def compare(A: MatrixEngine, B: MatrixEngine, tol=1e-9):
        """
        Analyzes the Matrix Inequality A ⪰ B in the Loewner Partial Order.
        A ⪰ B iff (A - B) is Positive Semidefinite.
        """
        if A.shape != B.shape:
            raise ValueError("Matrices must have the same dimensions for Loewner comparison.")
            
        mat_A = np.array(A.matrix, dtype=complex if np.iscomplexobj(A.matrix) else float)
        mat_B = np.array(B.matrix, dtype=complex if np.iscomplexobj(B.matrix) else float)
        
        Diff = MatrixEngine(mat_A - mat_B)
        
        if not DefinitenessClassifier.is_hermitian(Diff, tol):
            raise ValueError("Matrix inequality is defined only over Hermitian matrices.")
            
        report = DefinitenessClassifier.classify_via_eigenvalues(Diff, tol)
        
        # A ⪰ B if PD or PSD
        a_geq_b = report['status'] in ["Positive Definite (PD)", "Positive Semidefinite (PSD)", "Zero Matrix (Both PSD and NSD)"]
        # A ≻ B if strictly PD
        a_gt_b = report['status'] == "Positive Definite (PD)"
        
        return {
            "A_minus_B": Diff,
            "status": report['status'],
            "A_geq_B": a_geq_b,
            "A_gt_B": a_gt_b,
            "diff_eigenvalues": report['eigenvalues']
        }
