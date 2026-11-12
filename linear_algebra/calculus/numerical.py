import numpy as np

class FiniteDifferenceEngine:
    @staticmethod
    def gradient(func, x, h=1e-5, method='central'):
        """Computes the numerical gradient of a scalar function."""
        n = len(x)
        grad = np.zeros(n)
        for i in range(n):
            x_forward = np.array(x, dtype=float)
            x_backward = np.array(x, dtype=float)
            x_forward[i] += h
            x_backward[i] -= h
            
            if method == 'forward':
                grad[i] = (func(*x_forward) - func(*x)) / h
            elif method == 'backward':
                grad[i] = (func(*x) - func(*x_backward)) / h
            else: # central
                grad[i] = (func(*x_forward) - func(*x_backward)) / (2 * h)
        return grad

    @staticmethod
    def hessian(func, x, h=1e-4):
        """Computes the numerical Hessian matrix of a scalar function using central differences."""
        n = len(x)
        hess = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                x_ij1 = np.array(x, dtype=float); x_ij1[i] += h; x_ij1[j] += h
                x_ij2 = np.array(x, dtype=float); x_ij2[i] += h; x_ij2[j] -= h
                x_ij3 = np.array(x, dtype=float); x_ij3[i] -= h; x_ij3[j] += h
                x_ij4 = np.array(x, dtype=float); x_ij4[i] -= h; x_ij4[j] -= h
                
                term1 = func(*x_ij1) - func(*x_ij2)
                term2 = func(*x_ij3) - func(*x_ij4)
                hess[i, j] = (term1 - term2) / (4 * h * h)
                
        # Enforce symmetry mathematically guaranteed for smooth functions
        return (hess + hess.T) / 2
        
    @staticmethod
    def directional_derivative(func, x, v, h=1e-5):
        """Computes D_v f(x) = lim_{t->0} [f(x + tv) - f(x)] / t"""
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-15:
            return 0.0
        v_dir = np.array(v) / v_norm # Normalize direction
        
        x_forward = np.array(x) + h * v_dir
        x_backward = np.array(x) - h * v_dir
        return (func(*x_forward) - func(*x_backward)) / (2 * h)
