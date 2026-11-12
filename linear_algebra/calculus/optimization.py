import numpy as np

class OptimizationEngine:
    @staticmethod
    def gradient_descent(grad_func, x0, lr=0.01, max_iter=100, tol=1e-6):
        """Standard Gradient Descent."""
        x = np.array(x0, dtype=float)
        history = [x.copy()]
        for i in range(max_iter):
            g = grad_func(x)
            if np.linalg.norm(g) < tol:
                return x, history, True
            x = x - lr * g
            history.append(x.copy())
        return x, history, False

    @staticmethod
    def newtons_method(grad_func, hess_func, x0, max_iter=50, tol=1e-6):
        """Newton's Method for Optimization."""
        x = np.array(x0, dtype=float)
        history = [x.copy()]
        for i in range(max_iter):
            g = grad_func(x)
            if np.linalg.norm(g) < tol:
                return x, history, True
            H = hess_func(x)
            try:
                # x_{k+1} = x_k - H^{-1} g
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                # Fallback to gradient descent step if Hessian is singular
                step = 0.01 * g
            x = x - step
            history.append(x.copy())
        return x, history, False
