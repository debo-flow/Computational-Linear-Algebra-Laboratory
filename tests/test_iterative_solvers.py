import pytest
import numpy as np
import scipy.sparse as sps
from linear_algebra.iterative_solvers.generators import MatrixGenerators
from linear_algebra.iterative_solvers.solvers import IterativeSolvers
from linear_algebra.iterative_solvers.stationary import StationarySolvers
from linear_algebra.iterative_solvers.preconditioners import Preconditioners
from linear_algebra.iterative_solvers.krylov import KrylovSubspace

def test_poisson_generators():
    A_1d = MatrixGenerators.poisson_1d(10)
    assert A_1d.shape == (10, 10)
    # Check trace = 10 * 2 = 20
    assert np.isclose(A_1d.diagonal().sum(), 20.0)
    
    A_2d = MatrixGenerators.poisson_2d(3, 3)
    assert A_2d.shape == (9, 9)

def test_cg_convergence_spd():
    A = MatrixGenerators.poisson_1d(20)
    b = np.ones(20)
    x, info, hist = IterativeSolvers.cg(A, b, tol=1e-7)
    
    assert info == 0
    assert hist[-1] < 1e-7
    # Verify A@x = b
    assert np.linalg.norm(A @ x - b) / np.linalg.norm(b) < 1e-6

def test_gmres_nonsymmetric():
    # Construct an asymmetric sparse matrix
    n = 10
    A = sps.diags([-np.ones(n-1), 2*np.ones(n), 0.5*np.ones(n-1)], [-1, 0, 1], format='csr')
    b = np.ones(n)
    
    x, info, hist = IterativeSolvers.gmres(A, b, tol=1e-7)
    assert info == 0
    assert hist[-1] < 1e-7

def test_jacobi_convergence():
    # Strictly diagonally dominant matrix
    A = sps.diags([np.ones(9), 4*np.ones(10), np.ones(9)], [-1, 0, 1], format='csr')
    b = np.ones(10)
    
    x, info, hist = StationarySolvers.jacobi(A, b, tol=1e-5)
    assert info == 0
    assert hist[-1] < 1e-5

def test_preconditioner_ilu():
    A = MatrixGenerators.poisson_1d(20)
    b = np.ones(20)
    M = Preconditioners.ilu(A)
    
    x, info, hist = IterativeSolvers.gmres(A, b, tol=1e-7, M=M)
    assert info == 0
    # ILU on a 1D poisson is essentially exact, should converge in 1-2 iters
    assert len(hist) < 5

def test_arnoldi_iteration():
    A = np.diag([1, 2, 3, 4, 5])
    v = np.ones(5)
    Q, H = KrylovSubspace.arnoldi_iteration(A, v, k=3)
    
    # Q should be orthogonal
    ortho_err = np.linalg.norm(Q.T @ Q - np.eye(4))
    assert ortho_err < 1e-10
    # A V_k = V_{k+1} H_k
    recon_err = np.linalg.norm(A @ Q[:, :3] - Q @ H)
    assert recon_err < 1e-10
