import numpy as np
import sympy as sp
from linear_algebra.matrix.matrix_engine import MatrixEngine

class LinearSystem:
    def __init__(self, A: MatrixEngine, B: MatrixEngine):
        """Initializes the linear system Ax = B."""
        if A.rows != B.rows:
            raise ValueError(f"Dimension mismatch: Coefficient matrix A has {A.rows} equations, but right-hand side B has {B.rows}.")
        self.A = A
        self.B = B
        self.is_symbolic = A.is_symbolic or B.is_symbolic

    def augmented_matrix(self):
        """Returns the augmented matrix [A | B]."""
        if self.is_symbolic:
            return MatrixEngine(sp.Matrix(self.A.matrix).row_join(sp.Matrix(self.B.matrix)), True)
        return MatrixEngine(np.hstack((self.A.matrix, self.B.matrix)))
