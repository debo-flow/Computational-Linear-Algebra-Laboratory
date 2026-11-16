import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

class FiniteElementPDE:
    @staticmethod
    def solve_poisson_1d(mesh, f_rhs_func):
        """
        1D Finite Element Method (Galerkin) for -u'' = f(x)
        Uses piecewise linear basis functions.
        """
        nx, dx = mesh.nx, mesh.dx
        
        # Global Stiffness Matrix Assembly
        # For linear elements: local stiffness is [1, -1; -1, 1] / dx
        diagonals = [np.ones(nx-1) * (-1/dx), np.ones(nx) * (2/dx), np.ones(nx-1) * (-1/dx)]
        K = sps.diags(diagonals, [-1, 0, 1], format='lil')
        
        # Force Vector Assembly (Numerical integration via trapezoidal approximation)
        F = np.zeros(nx)
        for i in range(1, nx - 1):
            F[i] = f_rhs_func(mesh.x[i]) * dx
            
        # Apply Dirichlet Boundaries u(0) = 0, u(L) = 0
        K[0, :], K[-1, :] = 0, 0
        K[0, 0], K[-1, -1] = 1, 1
        F[0], F[-1] = 0, 0
        
        u = spla.spsolve(K.tocsr(), F)
        return u
