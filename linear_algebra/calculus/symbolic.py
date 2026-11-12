import sympy as sp
import numpy as np
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

class SymbolicCalculusEngine:
    @staticmethod
    def safe_parse(expr_str):
        """Safely parses a string into a SymPy expression, preventing arbitrary code execution."""
        transformations = (standard_transformations + (implicit_multiplication_application,))
        return parse_expr(expr_str, transformations=transformations, evaluate=False)

    @staticmethod
    def compute_gradient(expr_str, var_symbols):
        """Computes the symbolic gradient vector."""
        expr = SymbolicCalculusEngine.safe_parse(expr_str)
        grad = [sp.diff(expr, var) for var in var_symbols]
        return expr, sp.Matrix(grad)

    @staticmethod
    def compute_jacobian(expr_strs, var_symbols):
        """Computes the symbolic Jacobian matrix for a vector-valued function."""
        exprs = [SymbolicCalculusEngine.safe_parse(e) for e in expr_strs]
        funcs = sp.Matrix(exprs)
        jacobian = funcs.jacobian(var_symbols)
        return funcs, jacobian

    @staticmethod
    def compute_hessian(expr_str, var_symbols):
        """Computes the symbolic Hessian matrix."""
        expr = SymbolicCalculusEngine.safe_parse(expr_str)
        hessian = sp.hessian(expr, var_symbols)
        return expr, hessian

    @staticmethod
    def evaluate_sym(sym_matrix, var_symbols, point):
        """Evaluates a symbolic Matrix or Expr at a specific numeric point."""
        subs_dict = {var: val for var, val in zip(var_symbols, point)}
        # Evaluate and cast to float numpy array
        evaluated = sym_matrix.subs(subs_dict)
        return np.array(evaluated.tolist(), dtype=float)
