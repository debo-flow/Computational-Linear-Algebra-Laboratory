import numpy as np
import sympy as sp
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.vectors.vector_engine import VectorEngine

class EigenEngine:
    @staticmethod
    def compute_eigen_data(A: MatrixEngine, tol=1e-9):
        """
        Computes eigenvalues, algebraic multiplicities (AM), geometric multiplicities (GM),
        and eigenspace bases for a square matrix.
        """
        if not A.is_square():
            raise ValueError("Eigenvalues are mathematically defined only for square matrices.")

        report = []
        if A.is_symbolic:
            # SymPy provides exact eigenvects: list of (eigenvalue, algebraic_multiplicity, [eigenvectors])
            eigen_info = sp.Matrix(A.matrix).eigenvects()
            for val, am, vecs in eigen_info:
                basis = [VectorEngine(list(v), True) for v in vecs]
                gm = len(basis)
                report.append({
                    "eigenvalue": val,
                    "algebraic_multiplicity": am,
                    "geometric_multiplicity": gm,
                    "eigenvectors": basis,
                    "is_defective": gm < am
                })
        else:
            # NumPy numerical computation
            vals, vecs = np.linalg.eig(A.matrix)
            
            # Group numerical eigenvalues that are closer than the tolerance
            grouped = {}
            for i, val in enumerate(vals):
                # Rounding for stable clustering of floating point near-duplicates
                rounded_real = round(val.real, 6)
                rounded_imag = round(val.imag, 6)
                key = rounded_real if rounded_imag == 0 else complex(rounded_real, rounded_imag)
                
                if key not in grouped:
                    grouped[key] = {'exact_val': val, 'vecs': []}
                grouped[key]['vecs'].append(vecs[:, i])
                
            for k, data in grouped.items():
                val = data['exact_val']
                v_list = data['vecs']
                am = len(v_list)
                
                # GM = Nullity(A - lambda * I) = n - Rank(A - lambda * I)
                I = np.eye(A.rows)
                shifted_A = A.matrix - val * I
                rank = np.linalg.matrix_rank(shifted_A, tol=tol)
                gm = A.rows - rank
                
                # Enforce mathematically guaranteed bounds despite floating point drift
                if gm > am: gm = am 
                if gm < 1: gm = 1
                
                basis = [VectorEngine(list(v)) for v in v_list]
                report.append({
                    "eigenvalue": val,
                    "algebraic_multiplicity": am,
                    "geometric_multiplicity": gm,
                    "eigenvectors": basis,
                    "is_defective": gm < am
                })
                
        return report
