import numpy as np
import scipy.integrate as spi

class ODESolverIVP:
    @staticmethod
    def euler_method(f, t_span, y0, n_steps):
        """Explicit Euler Method for Systems of ODEs."""
        t0, tf = t_span
        t = np.linspace(t0, tf, n_steps + 1)
        h = (tf - t0) / n_steps
        y = np.zeros((n_steps + 1, len(y0)))
        y[0] = y0
        
        for i in range(n_steps):
            y[i+1] = y[i] + h * np.array(f(t[i], y[i]))
        return t, y

    @staticmethod
    def rk4_method(f, t_span, y0, n_steps):
        """Classical Runge-Kutta 4th Order (RK4)."""
        t0, tf = t_span
        t = np.linspace(t0, tf, n_steps + 1)
        h = (tf - t0) / n_steps
        y = np.zeros((n_steps + 1, len(y0)))
        y[0] = y0
        
        for i in range(n_steps):
            ti, yi = t[i], y[i]
            k1 = np.array(f(ti, yi))
            k2 = np.array(f(ti + h/2, yi + h/2 * k1))
            k3 = np.array(f(ti + h/2, yi + h/2 * k2))
            k4 = np.array(f(ti + h, yi + h * k3))
            
            y[i+1] = yi + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
        return t, y

    @staticmethod
    def adaptive_solve(f, t_span, y0, method='RK45', atol=1e-6, rtol=1e-3):
        """
        Adaptive Step-Size Solvers (SciPy Backend).
        Supports RK45, RK23, DOP853, BDF (Stiff), Radau (Stiff).
        """
        sol = spi.solve_ivp(f, t_span, y0, method=method, atol=atol, rtol=rtol)
        if not sol.success:
            raise RuntimeError(f"ODE Solver failed: {sol.message}")
        return sol.t, sol.y.T, sol.nfev
