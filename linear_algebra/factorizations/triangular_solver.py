import numpy as np
import scipy.linalg as spla
from linear_algebra.matrix.matrix_engine import MatrixEngine

class TriangularSolver:
    @staticmethod
    def forward_substitution(L: MatrixEngine, b: MatrixEngine, unit_diagonal=False):
        """Solves L y = b for lower triangular L."""
        L_mat = np.array(L.matrix, dtype=complex if np.iscomplexobj(L.matrix) else float)
        b_mat = np.array(b.matrix, dtype=complex if np.iscomplexobj(b.matrix) else float)
        y = spla.solve_triangular(L_mat, b_mat, lower=True, unit_diagonal=unit_diagonal)
        return MatrixEngine(y)

    @staticmethod
    def backward_substitution(U: MatrixEngine, y: MatrixEngine, unit_diagonal=False):
        """Solves U x = y for upper triangular U."""
        U_mat = np.array(U.matrix, dtype=complex if np.iscomplexobj(U.matrix) else float)
        y_mat = np.array(y.matrix, dtype=complex if np.iscomplexobj(y.matrix) else float)
        x = spla.solve_triangular(U_mat, y_mat, lower=False, unit_diagonal=unit_diagonal)
        return MatrixEngine(x)

    @staticmethod
    def solve_via_lu(P: MatrixEngine, L: MatrixEngine, U: MatrixEngine, b: MatrixEngine):
        """
        Solves PA = LU => LUx = Pb.
        Step 1: y = L^(-1) P b
        Step 2: x = U^(-1) y
        """
        P_mat = np.array(P.matrix)
        b_mat = np.array(b.matrix, dtype=complex if np.iscomplexobj(b.matrix) else float)
        
        Pb = MatrixEngine(P_mat @ b_mat)
        y = TriangularSolver.forward_substitution(L, Pb, unit_diagonal=True) # L in PA=LU has unit diag
        x = TriangularSolver.backward_substitution(U, y)
        
        return x, y, Pb
