import numpy as np
import scipy.linalg as spla
import sympy as sp
from linear_algebra.matrix.matrix_engine import MatrixEngine
from .system_model import LinearSystem

class SystemSolvers:
    @staticmethod
    def direct_solve(system: LinearSystem):
        if system.A.rows != system.A.cols:
            raise ValueError("Direct solve requires a square coefficient matrix.")
        if system.is_symbolic:
            res = sp.Matrix(system.A.matrix).solve(sp.Matrix(system.B.matrix))
            return MatrixEngine(res, True)
        return MatrixEngine(np.linalg.solve(system.A.matrix, system.B.matrix))

    @staticmethod
    def least_squares(system: LinearSystem):
        if system.is_symbolic:
            raise NotImplementedError("Least squares is currently supported for numerical matrices only.")
        x, residuals, rank, s = np.linalg.lstsq(system.A.matrix, system.B.matrix, rcond=None)
        return MatrixEngine(x), residuals, rank, s

    @staticmethod
    def parametric_solve(system: LinearSystem):
        """Returns symbolic parameterized solution for systems with infinitely many solutions."""
        sym_A = sp.Matrix(system.A.matrix)
        sym_B = sp.Matrix(system.B.matrix)
        # linsolve handles both unique and infinite cases symbolically
        solution_set = sp.linsolve((sym_A, sym_B))
        if not solution_set:
            raise ValueError("System is inconsistent. No parametric solution exists.")
        
        # Convert finite set representation to a column vector MatrixEngine
        sol_tuple = list(solution_set)[0]
        sol_matrix = sp.Matrix(sol_tuple).T
        return MatrixEngine(sol_matrix, True)
