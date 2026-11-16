import pytest
import numpy as np
from linear_algebra.pde.mesh import Mesh1D, Mesh2D
from linear_algebra.pde.fdm import FiniteDifferencePDE
from linear_algebra.pde.fem import FiniteElementPDE
from linear_algebra.pde.fvm import FiniteVolumePDE
from linear_algebra.pde.spectral import SpectralMethods

def test_poisson_2d_fdm():
    mesh = Mesh2D(1.0, 1.0, 10, 10)
    # Zero forcing and Zero BC implies u = 0 everywhere
    def f_rhs(x, y): return 0.0
    def bc(x, y): return 0.0
    
    u = FiniteDifferencePDE.solve_poisson_2d(mesh, f_rhs, bc)
    assert np.allclose(u, 0.0)

def test_wave_equation_1d_cfl_violation():
    mesh = Mesh1D(1.0, 10) # dx = 0.1
    c = 1.0
    # dt = T / nt -> 2.0 / 10 = 0.2. CFL = c * dt / dx = 1.0 * 0.2 / 0.1 = 2.0 > 1.0
    
    def u0(x): return np.zeros_like(x)
    def v0(x): return np.zeros_like(x)
    
    with pytest.raises(ValueError, match="CFL number"):
        FiniteDifferencePDE.solve_wave_1d(mesh, 2.0, 10, c, u0, v0)

def test_fem_poisson_1d():
    mesh = Mesh1D(1.0, 11)
    # -u'' = 2 => u(x) = x(1 - x) is the analytic solution with u(0)=u(1)=0
    def f_rhs(x): return 2.0
    
    u = FiniteElementPDE.solve_poisson_1d(mesh, f_rhs)
    u_analytic = mesh.x * (1 - mesh.x)
    
    # Check max error
    assert np.max(np.abs(u - u_analytic)) < 1e-2

def test_spectral_fourier_derivative():
    # Domain [0, 2pi)
    x = np.linspace(0, 2*np.pi, 32, endpoint=False)
    u = np.sin(x)
    
    du_num = SpectralMethods.fourier_derivative_1d(u, 2*np.pi)
    du_analytic = np.cos(x)
    
    assert np.allclose(du_num, du_analytic)

def test_fvm_conservation():
    mesh = Mesh1D(1.0, 20)
    def u0(x): return np.ones_like(x)
    
    # Constant velocity advection over periodic boundaries must conserve total sum
    u = FiniteVolumePDE.solve_advection_1d(mesh, 1.0, 50, 0.5, u0)
    
    initial_mass = np.sum(u[0, :])
    final_mass = np.sum(u[-1, :])
    
    assert np.isclose(initial_mass, final_mass)
