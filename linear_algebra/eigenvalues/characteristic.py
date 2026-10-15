import numpy as np
import sympy as sp
from linear_algebra.matrix.matrix_engine import MatrixEngine

class CharacteristicPolynomial:
    @staticmethod
    def compute(A: MatrixEngine):
        """
        Computes the characteristic polynomial p(lambda) = det(lambda I - A)
        and extracts its roots.
        """
        if not A.is_square():
            raise ValueError("Characteristic polynomial requires a square matrix.")
            
        if A.is_symbolic:
            lam = sp.Symbol('lambda')
            poly = sp.Matrix(A.matrix).charpoly(lam)
            expr = poly.as_expr()
            roots = poly.all_roots()
            return expr, roots, True
        else:
            # np.poly returns the coefficients [c_n, c_{n-1}, ..., c_0] of the characteristic polynomial
            coeffs = np.poly(A.matrix)
            roots = np.roots(coeffs)
            return coeffs, roots, False
            
    @staticmethod
    def format_numeric_poly(coeffs):
        """Formats numpy polynomial coefficients into a readable LaTeX string."""
        degree = len(coeffs) - 1
        terms = []
        for i, c in enumerate(coeffs):
            if abs(c) < 1e-9: continue
            power = degree - i
            term_str = f"{c:.4g}"
            if power == 0:
                terms.append(term_str)
            elif power == 1:
                terms.append(f"{term_str}\\lambda")
            else:
                terms.append(f"{term_str}\\lambda^{{{power}}}")
        return " + ".join(terms).replace("+ -", "- ")
