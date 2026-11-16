import pytest
import numpy as np
from linear_algebra.integration_odes.integration import NumericalIntegration
from linear_algebra.integration_odes.ode_ivp import ODESolverIVP
from linear_algebra.integration_odes.ode_bvp import ODESolverBVP

def test_trapezoidal_rule():
    # Integral of x^2 from 0 to 1 is 1/3
    f = lambda x: x**2
    val, _, _ = NumericalIntegration.trapezoidal_rule(f, 0, 1, 100)
    assert np.isclose(val, 1/3, atol=1e-3)

def test_simpsons_rule():
    f = lambda x: x**2
    # Simpson's 1/3 rule is exact for quadratics
    val, _, _ = NumericalIntegration.simpsons_rule(f, 0, 1, 2)
    assert np.isclose(val, 1/3, atol=1e-12)

def test_gauss_legendre():
    # Polynomial of degree 5 -> exact with 3 Gauss nodes, we use 5 to be safe
    f = lambda x: x**5 - 2*x**3 + 1
    # Int(0 to 1) = 1/6 - 2/4 + 1 = 1/6 - 1/2 + 1 = 2/3
    val, _, _ = NumericalIntegration.gauss_legendre(f, 0, 1, deg=5)
    assert np.isclose(val, 2/3, atol=1e-12)

def test_rk4_method():
    # dy/dt = -y, y(0) = 1 -> y(t) = e^-t
    f = lambda t, y: [-y[0]]
    t, y = ODESolverIVP.rk4_method(f, (0, 1), [1.0], 100)
    
    y_final = y[-1, 0]
    expected = np.exp(-1.0)
    assert np.isclose(y_final, expected, atol=1e-6)

def test_adaptive_solve_scipy():
    f = lambda t, y: [y[1], -y[0]] # Harmonic oscillator
    t, y, nfev = ODESolverIVP.adaptive_solve(f, (0, 2*np.pi), [1.0, 0.0], method='RK45')
    
    # At t=2pi, y[0] should be cos(2pi)=1, y[1] should be -sin(2pi)=0
    assert np.isclose(y[-1, 0], 1.0, atol=1e-3)
    assert np.isclose(y[-1, 1], 0.0, atol=1e-3)

def test_bvp_collocation():
    # y'' = -y, y(0)=0, y(pi/2)=1 => y(x) = sin(x)
    def f_ode(x, y):
        return np.vstack((y[1], -y[0]))
    def bc(ya, yb):
        return np.array([ya[0], yb[0] - 1])
        
    x = np.linspace(0, np.pi/2, 10)
    y_guess = np.zeros((2, x.size))
    
    x_sol, y_sol, res = ODESolverBVP.solve_2nd_order_bvp(f_ode, bc, x, y_guess)
    
    # Check y(pi/4) approx sin(pi/4) = 0.7071
    idx = np.argmin(np.abs(x_sol - np.pi/4))
    assert np.isclose(y_sol[0, idx], np.sin(x_sol[idx]), atol=1e-3)
    assert np.max(res) < 1e-3
