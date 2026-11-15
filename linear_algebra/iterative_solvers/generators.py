import numpy as np
import scipy.sparse as sps

class MatrixGenerators:
    @staticmethod
    def poisson_1d(n: int):
        """Generates a 1D Poisson (Laplacian) matrix (Symmetric Positive-Definite)."""
        diagonals = [-np.ones(n-1), 2 * np.ones(n), -np.ones(n-1)]
        return sps.diags(diagonals, [-1, 0, 1], format='csr')

    @staticmethod
    def poisson_2d(nx: int, ny: int):
        """Generates a 2D Poisson matrix using Kronecker sums."""
        Dx = MatrixGenerators.poisson_1d(nx)
        Dy = MatrixGenerators.poisson_1d(ny)
        Ix = sps.eye(nx, format='csr')
        Iy = sps.eye(ny, format='csr')
        # A = Iy (x) Dx + Dy (x) Ix
        return sps.kronsum(Dy, Dx, format='csr')

    @staticmethod
    def random_sparse_spd(n: int, density: float = 0.1, seed: int = 42):
        """Generates a random Sparse Symmetric Positive-Definite matrix."""
        np.random.seed(seed)
        import scipy.sparse as sps
        A = sps.random(n, n, density=density, format='csr', data_rvs=np.random.randn)
        A = A.T @ A + 0.1 * sps.eye(n)
        return A
