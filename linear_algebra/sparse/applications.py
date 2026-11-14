import numpy as np
import scipy.sparse as sps

class SparseApplications:
    @staticmethod
    def laplacian_1d(n):
        """Creates a 1D Laplacian matrix (Finite Difference) natively in CSR format."""
        diags = [np.ones(n)*-1, np.ones(n)*2, np.ones(n)*-1]
        mat = sps.spdiags(diags, [-1, 0, 1], n, n, format='csr')
        return mat

    @staticmethod
    def laplacian_2d(nx, ny):
        """Creates a 2D Poisson/Laplacian matrix via Kronecker tensor products."""
        # 1D operators
        Dx = SparseApplications.laplacian_1d(nx)
        Dy = SparseApplications.laplacian_1d(ny)
        
        # Identity matrices
        Ix = sps.eye(nx)
        Iy = sps.eye(ny)
        
        # 2D Laplacian = D_x ⊗ I_y + I_x ⊗ D_y
        L = sps.kronsum(Dy, Dx, format='csr')
        return L
