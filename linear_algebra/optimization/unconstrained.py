import numpy as np

class UnconstrainedOptimization:
    @staticmethod
    def backtracking_line_search(func, x, p, grad, alpha=1.0, rho=0.5, c=1e-4):
        """Armijo condition backtracking line search."""
        f_x, _, _ = func(x)
        while True:
            f_new, _, _ = func(x + alpha * p)
            if f_new <= f_x + c * alpha * np.dot(grad, p):
                break
            alpha *= rho
            if alpha < 1e-10:
                break
        return alpha

    @staticmethod
    def gradient_descent(func, x0, lr=0.01, max_iter=1000, tol=1e-6, use_line_search=False):
        x = np.array(x0, dtype=float)
        history = [x.copy()]
        for i in range(max_iter):
            _, grad, _ = func(x)
            if np.linalg.norm(grad) < tol:
                return x, history, True
            
            step = lr
            if use_line_search:
                step = UnconstrainedOptimization.backtracking_line_search(func, x, -grad, grad)
                
            x = x - step * grad
            history.append(x.copy())
        return x, history, False

    @staticmethod
    def adam(func, x0, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iter=1000, tol=1e-6):
        x = np.array(x0, dtype=float)
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        history = [x.copy()]
        
        for t in range(1, max_iter + 1):
            _, grad, _ = func(x)
            if np.linalg.norm(grad) < tol:
                return x, history, True
                
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            
            x = x - lr * m_hat / (np.sqrt(v_hat) + epsilon)
            history.append(x.copy())
            
        return x, history, False

    @staticmethod
    def newtons_method(func, x0, max_iter=100, tol=1e-6):
        x = np.array(x0, dtype=float)
        history = [x.copy()]
        
        for i in range(max_iter):
            _, grad, hess = func(x)
            if np.linalg.norm(grad) < tol:
                return x, history, True
                
            try:
                # Solve H p = -g to avoid explicit inversion
                p = np.linalg.solve(hess, -grad)
            except np.linalg.LinAlgError:
                # Fallback to steepest descent direction if Hessian is singular
                p = -grad
                
            alpha = UnconstrainedOptimization.backtracking_line_search(func, x, p, grad)
            x = x + alpha * p
            history.append(x.copy())
            
        return x, history, False

    @staticmethod
    def bfgs(func, x0, max_iter=100, tol=1e-6):
        """Quasi-Newton BFGS Method."""
        x = np.array(x0, dtype=float)
        n = len(x)
        H_inv = np.eye(n) # Approximate inverse Hessian
        history = [x.copy()]
        
        for i in range(max_iter):
            _, grad, _ = func(x)
            if np.linalg.norm(grad) < tol:
                return x, history, True
                
            p = -H_inv @ grad
            alpha = UnconstrainedOptimization.backtracking_line_search(func, x, p, grad)
            
            x_next = x + alpha * p
            _, grad_next, _ = func(x_next)
            
            s = x_next - x
            y = grad_next - grad
            
            # BFGS Update
            rho = 1.0 / (np.dot(y, s) + 1e-10)
            I = np.eye(n)
            H_inv = (I - rho * np.outer(s, y)) @ H_inv @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
            
            x = x_next
            history.append(x.copy())
            
        return x, history, False
