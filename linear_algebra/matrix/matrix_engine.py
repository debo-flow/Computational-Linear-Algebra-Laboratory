import numpy as np
import sympy as sp

class MatrixEngine:
    def __init__(self, data, is_symbolic=False):
        """Initialize the matrix with numerical or symbolic data."""
        self.is_symbolic = is_symbolic
        
        try:
            if is_symbolic:
                self.matrix = sp.Matrix(data)
                self.shape = self.matrix.shape
            else:
                self.matrix = np.array(data, dtype=float)
                if self.matrix.ndim != 2:
                    raise ValueError("Input must be a 2D array or list of lists.")
                self.shape = self.matrix.shape
        except Exception as e:
            raise ValueError(f"Invalid matrix input: {e}")

    @property
    def rows(self):
        return self.shape[0]

    @property
    def cols(self):
        return self.shape[1]

    # --- Properties Classification ---
    def is_square(self):
        return self.rows == self.cols

    def is_row_matrix(self):
        return self.rows == 1

    def is_column_matrix(self):
        return self.cols == 1

    def is_zero_matrix(self, tol=1e-9):
        if self.is_symbolic:
            return self.matrix.is_zero_matrix
        return np.allclose(self.matrix, 0, atol=tol)

    def is_identity(self, tol=1e-9):
        if not self.is_square(): return False
        if self.is_symbolic:
            return self.matrix == sp.eye(self.rows)
        return np.allclose(self.matrix, np.eye(self.rows), atol=tol)

    def is_diagonal(self, tol=1e-9):
        if not self.is_square(): return False
        if self.is_symbolic:
            return self.matrix.is_diagonal()
        off_diag = self.matrix - np.diag(np.diagonal(self.matrix))
        return np.allclose(off_diag, 0, atol=tol)

    def is_symmetric(self, tol=1e-9):
        if not self.is_square(): return False
        if self.is_symbolic:
            return self.matrix.is_symmetric()
        return np.allclose(self.matrix, self.matrix.T, atol=tol)
    
    def get_properties_report(self):
        return {
            "Dimensions": f"{self.rows} × {self.cols}",
            "Square": self.is_square(),
            "Row Matrix": self.is_row_matrix(),
            "Column Matrix": self.is_column_matrix(),
            "Zero Matrix": self.is_zero_matrix(),
            "Identity Matrix": self.is_identity(),
            "Diagonal Matrix": self.is_diagonal(),
            "Symmetric": self.is_symmetric()
        }

    # --- Generators ---
    @staticmethod
    def generate_identity(n, symbolic=False):
        return MatrixEngine(sp.eye(n) if symbolic else np.eye(n), symbolic)

    @staticmethod
    def generate_zero(m, n, symbolic=False):
        return MatrixEngine(sp.zeros(m, n) if symbolic else np.zeros((m, n)), symbolic)

    def to_latex(self):
        if self.is_symbolic:
            return sp.latex(self.matrix)
        
        # Format NumPy array as LaTeX bmatrix
        lines = []
        for row in self.matrix:
            lines.append(" & ".join([f"{val:g}" for val in row]))
        return "\\begin{bmatrix}\n" + " \\\\\n".join(lines) + "\n\\end{bmatrix}"
