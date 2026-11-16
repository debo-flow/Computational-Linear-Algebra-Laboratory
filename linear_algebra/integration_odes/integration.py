import numpy as np
import scipy.integrate as spi

class NumericalIntegration:
    @staticmethod
    def trapezoidal_rule(f, a, b, n):
        """Composite Trapezoidal Rule."""
        if n < 1: raise ValueError("Number of subintervals n must be >= 1.")
        x = np.linspace(a, b, n + 1)
        y = f(x)
        h = (b - a) / n
        integral = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
        return integral, x, y

    @staticmethod
    def simpsons_rule(f, a, b, n):
        """Composite Simpson's 1/3 Rule."""
        if n < 2 or n % 2 != 0: raise ValueError("n must be an even integer >= 2 for Simpson's 1/3 rule.")
        x = np.linspace(a, b, n + 1)
        y = f(x)
        h = (b - a) / n
        integral = (h / 3) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])
        return integral, x, y

    @staticmethod
    def gauss_legendre(f, a, b, deg):
        """Gauss-Legendre Quadrature."""
        if deg < 1: raise ValueError("Degree must be >= 1.")
        nodes, weights = np.polynomial.legendre.leggauss(deg)
        # Map nodes from [-1, 1] to [a, b]
        t = 0.5 * (x := (b - a) * nodes + (a + b))
        integral = 0.5 * (b - a) * np.sum(weights * f(t))
        return integral, t, weights

    @staticmethod
    def adaptive_quadrature(f, a, b, epsabs=1e-8, epsrel=1e-8):
        """Adaptive Quadrature (via SciPy QUADPACK)."""
        integral, abserr, info = spi.quad(f, a, b, epsabs=epsabs, epsrel=epsrel, full_output=True)
        return integral, abserr, info['neval']

    @staticmethod
    def monte_carlo_1d(f, a, b, num_samples, seed=42):
        """1D Monte Carlo Integration."""
        np.random.seed(seed)
        samples = np.random.uniform(a, b, num_samples)
        f_samples = f(samples)
        
        integral = (b - a) * np.mean(f_samples)
        variance = np.var(f_samples)
        std_error = (b - a) * np.sqrt(variance / num_samples)
        
        return integral, std_error, samples, f_samples
