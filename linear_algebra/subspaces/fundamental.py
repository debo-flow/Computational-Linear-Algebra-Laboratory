import numpy as np
import scipy.linalg as spla
import sympy as sp
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.matrix.operations import MatrixOperations
from linear_algebra.vectors.vector_engine import VectorEngine
from linear_algebra.vector_spaces.basis import BasisAnalyzer

class FundamentalSubspaces:
    @staticmethod
    def get_column_space(A: MatrixEngine):
        """Returns basis for Col(A) using ORIGINAL columns of A."""
        _, pivots = MatrixOperations.rref(A)
        col_basis = []
        for p in pivots:
            if A.is_symbolic:
                col_basis.append(VectorEngine(list(A.matrix.col(p)), True))
            else:
                col_basis.append(VectorEngine(list(A.matrix[:, p])))
        return col_basis, pivots

    @staticmethod
    def get_row_space(A: MatrixEngine):
        """Returns basis for Row(A) derived from RREF nonzero rows."""
        return BasisAnalyzer.get_row_space_basis(A)

    @staticmethod
    def get_null_space(A: MatrixEngine, tol=1e-9):
        """Returns basis for Null(A). Uses SymPy exact for symbolic, SciPy for numeric stability."""
        if A.is_symbolic:
            sym_A = sp.Matrix(A.matrix)
            null_basis = sym_A.nullspace()
            return [VectorEngine(list(n), True) for n in null_basis]
        
        # Numeric: SciPy null_space utilizes SVD and provides an orthonormal basis
        ns = spla.null_space(A.matrix, rcond=tol)
        if ns.size == 0:
            return []
        # Return columns as vectors
        return [VectorEngine(list(ns[:, i])) for i in range(ns.shape[1])]

    @staticmethod
    def get_left_null_space(A: MatrixEngine, tol=1e-9):
        """Returns basis for Null(A^H), the Left Null Space."""
        if A.is_symbolic:
            A_H = MatrixEngine(sp.Matrix(A.matrix).H, True)
        else:
            A_H = MatrixEngine(A.matrix.conj().T)
            
        return FundamentalSubspaces.get_null_space(A_H, tol)

    @staticmethod
    def analyze(A: MatrixEngine, tol=1e-9):
        """Comprehensive Fundamental Subspace analysis."""
        m, n = A.rows, A.cols
        rank = MatrixOperations.rank(A, tol)
        
        col_basis, pivots = FundamentalSubspaces.get_column_space(A)
        row_basis = FundamentalSubspaces.get_row_space(A)
        null_basis = FundamentalSubspaces.get_null_space(A, tol)
        left_null_basis = FundamentalSubspaces.get_left_null_space(A, tol)
        
        # Verify Rank-Nullity
        nullity = len(null_basis)
        left_nullity = len(left_null_basis)
        
        # Verification Residuals: A * v_null = 0
        null_residual = 0.0
        if not A.is_symbolic and null_basis:
            res_norms = [np.linalg.norm(A.matrix @ np.array(v.vector)) for v in null_basis]
            null_residual = max(res_norms)
            
        return {
            "dimensions": (m, n),
            "rank": rank,
            "nullity": nullity,
            "left_nullity": left_nullity,
            "col_space": {"basis": col_basis, "dim": len(col_basis), "pivots": pivots},
            "row_space": {"basis": row_basis, "dim": len(row_basis)},
            "null_space": {"basis": null_basis, "dim": nullity, "max_residual": null_residual},
            "left_null_space": {"basis": left_null_basis, "dim": left_nullity},
            "rank_nullity_domain": rank + nullity == n,
            "rank_nullity_codomain": rank + left_nullity == m
        }
