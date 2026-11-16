import numpy as np

class NonlinearLeastSquares:
    @staticmethod
    def gauss_newton(res_func, jac_func, x0, max_iter=100, tol=1e-6):
        """Gauss-Newton Method for Nonlinear Least Squares."""
        x = np.array(x0, dtype=float)
        history = [x.copy()]
        
        for i in range(max_iter):
            r = res_func(x)
            J = jac_func(x)
            grad = J.T @ r
            
            if np.linalg.norm(grad) < tol:
                return x, history, True
                
            try:
                # Solve (J^T J) p = -J^T r
                p = np.linalg.solve(J.T @ J, -grad)
            except np.linalg.LinAlgError:
                return x, history, False
                
            x = x + p
            history.append(x.copy())
            
        return x, history, False

    @staticmethod
    def levenberg_marquardt(res_func, jac_func, x0, lmbda=0.01, max_iter=100, tol=1e-6):
        """Levenberg-Marquardt Method (Damped Gauss-Newton)."""
        x = np.array(x0, dtype=float)
        history = [x.copy()]
        
        for i in range(max_iter):
            r = res_func(x)
            J = jac_func(x)
            grad = J.T @ r
            
            if np.linalg.norm(grad) < tol:
                return x, history, True
                
            # (J^T J + lambda I) p = -J^T r
            H_approx = J.T @ J
            H_damped = H_approx + lmbda * np.eye(H_approx.shape[0])
            
            try:
                p = np.linalg.solve(H_damped, -grad)
            except np.linalg.LinAlgError:
                return x, history, False
                
            # Trust region update heuristic
            x_new = x + p
            r_new = res_func(x_new)
            
            if np.sum(r_new**2) < np.sum(r**2):
                x = x_new
                lmbda /= 10 # Step successful, reduce damping
            else:
                lmbda *= 10 # Step failed, increase damping
                
            history.append(x.copy())
            
        return x, history, False
