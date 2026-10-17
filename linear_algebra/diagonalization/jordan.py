import sympy as sp
from linear_algebra.matrix.matrix_engine import MatrixEngine

class JordanCanonicalForm:
    @staticmethod
    def compute(A: MatrixEngine):
        """
        Extracts the Jordan Canonical Form A = P J P^(-1).
        Because JCF is highly unstable to perturbations, it strictly requires symbolic matrices.
        """
        if not A.is_symbolic:
            raise ValueError("Jordan canonical form is highly sensitive to perturbations and mathematically requires exact symbolic arithmetic.")
        
        sym_A = sp.Matrix(A.matrix)
        P, J = sym_A.jordan_form()
        
        return MatrixEngine(P, True), MatrixEngine(J, True)
