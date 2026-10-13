import sympy as sp
from linear_algebra.matrix.matrix_engine import MatrixEngine
from .system_model import LinearSystem

class EliminationEngine:
    @staticmethod
    def gaussian_elimination_steps(system: LinearSystem):
        """Returns a list of state dicts showing step-by-step row operations."""
        # Using SymPy exact arithmetic to avoid floating-point mess in educational views
        aug = sp.Matrix(system.augmented_matrix().matrix)
        m, n = aug.shape
        steps = [{"description": "Initial Augmented Matrix $[A|B]$", "matrix": MatrixEngine(aug, True)}]
        
        row, col = 0, 0
        while row < m and col < n - system.B.cols:
            # 1. Partial Pivoting
            pivot_row = row
            for i in range(row + 1, m):
                if abs(aug[i, col]) > abs(aug[pivot_row, col]):
                    pivot_row = i
            
            if aug[pivot_row, col] == 0:
                col += 1
                continue
                
            # 2. Row Swap
            if pivot_row != row:
                aug.row_swap(row, pivot_row)
                steps.append({"description": f"Swap $R_{row+1} \\leftrightarrow R_{pivot_row+1}$", "matrix": MatrixEngine(aug, True)})
                
            # 3. Scale Pivot
            pivot_val = aug[row, col]
            if pivot_val != 1:
                # SymPy row multiplication
                aug[row, :] = aug[row, :] / pivot_val
                steps.append({"description": f"Scale $R_{row+1} \\to \\frac{{1}}{{{pivot_val}}} R_{row+1}$", "matrix": MatrixEngine(aug, True)})
                
            # 4. Eliminate Below
            for i in range(row + 1, m):
                factor = aug[i, col]
                if factor != 0:
                    aug[i, :] = aug[i, :] - factor * aug[row, :]
                    steps.append({"description": f"Eliminate $R_{i+1} \\to R_{i+1} - ({factor})R_{row+1}$", "matrix": MatrixEngine(aug, True)})
                    
            row += 1
            col += 1
            
        return steps
