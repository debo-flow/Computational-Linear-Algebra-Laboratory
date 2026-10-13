import numpy as np
import sympy as sp
from .matrix_engine import MatrixEngine
from .operations import MatrixOperations

class SingularityAnalyzer:
    @staticmethod
    def analyze(A: MatrixEngine, tol=1e-9):
        """Analyzes exact and numerical singularity, rank, and condition."""
        report = {
            "is_square": A.is_square(),
            "dimensions": f"{A.rows} × {A.cols}",
            "rank": MatrixOperations.rank(A, tol)
        }
        
        if A.is_square():
            report["determinant"] = MatrixOperations.determinant(A, tol)
            
            if A.is_symbolic:
                report["is_singular"] = sp.simplify(report["determinant"]) == 0
                report["status"] = "Singular (Exact)" if report["is_singular"] else "Non-Singular (Exact)"
            else:
                svd_vals = np.linalg.svd(A.matrix, compute_uv=False)
                report["singular_values"] = svd_vals
                min_sv = svd_vals[-1]
                max_sv = svd_vals[0]
                
                if min_sv < tol:
                    report["is_singular"] = True
                    report["status"] = "Numerically Singular (Rank Deficient)"
                else:
                    cond = max_sv / min_sv
                    report["condition_number"] = cond
                    if cond > (1 / tol):
                        report["is_singular"] = False
                        report["status"] = "Ill-Conditioned (Near-Singular)"
                    else:
                        report["is_singular"] = False
                        report["status"] = "Non-Singular (Well-Conditioned)"
        else:
            report["status"] = "Not Applicable (Rectangular Matrix)"
            if not A.is_symbolic:
                report["singular_values"] = np.linalg.svd(A.matrix, compute_uv=False)
                
        return report
