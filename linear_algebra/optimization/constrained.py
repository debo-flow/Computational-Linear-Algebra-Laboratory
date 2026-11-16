import numpy as np
from .unconstrained import UnconstrainedOptimization

class ConstrainedOptimization:
    @staticmethod
    def quadratic_penalty_method(func, eq_funcs, ineq_funcs, x0, mu=1.0, max_iter=500):
        """
        Penalty method for Constrained Optimization.
        minimize f(x) + mu/2 * sum(h(x)^2) + mu/2 * sum(max(0, g(x))^2)
        """
        def penalized_func(x):
            f_val, f_grad, f_hess = func(x)
            
            p_val = f_val
            p_grad = f_grad.copy()
            p_hess = f_hess.copy()
            
            # Equality constraints: h(x) = 0
            for h_func in eq_funcs:
                h_val, h_grad, h_hess = h_func(x)
                p_val += (mu / 2) * (h_val ** 2)
                p_grad += mu * h_val * h_grad
                p_hess += mu * (np.outer(h_grad, h_grad) + h_val * h_hess)
                
            # Inequality constraints: g(x) <= 0
            for g_func in ineq_funcs:
                g_val, g_grad, g_hess = g_func(x)
                if g_val > 0:
                    p_val += (mu / 2) * (g_val ** 2)
                    p_grad += mu * g_val * g_grad
                    p_hess += mu * (np.outer(g_grad, g_grad) + g_val * g_hess)
                    
            return p_val, p_grad, p_hess

        # Solve unconstrained problem using BFGS
        x_opt, history, conv = UnconstrainedOptimization.bfgs(penalized_func, x0, max_iter=max_iter)
        return x_opt, history, conv

    @staticmethod
    def check_kkt_conditions(func, eq_funcs, ineq_funcs, x, lambda_eq, mu_ineq, tol=1e-5):
        """
        Evaluates the Karush-Kuhn-Tucker (KKT) conditions.
        L(x, lambda, mu) = f(x) + sum(lambda * h(x)) + sum(mu * g(x))
        """
        _, f_grad, _ = func(x)
        
        stationarity_grad = f_grad.copy()
        primal_feasibility = True
        dual_feasibility = True
        complementary_slackness = True
        
        # Equalities
        for i, h_func in enumerate(eq_funcs):
            h_val, h_grad, _ = h_func(x)
            stationarity_grad += lambda_eq[i] * h_grad
            if abs(h_val) > tol: primal_feasibility = False
            
        # Inequalities
        for j, g_func in enumerate(ineq_funcs):
            g_val, g_grad, _ = g_func(x)
            stationarity_grad += mu_ineq[j] * g_grad
            
            if g_val > tol: primal_feasibility = False
            if mu_ineq[j] < -tol: dual_feasibility = False # mu must be >= 0
            if abs(mu_ineq[j] * g_val) > tol: complementary_slackness = False
            
        stationarity = np.linalg.norm(stationarity_grad) < tol
        
        return {
            "Stationarity (∇L=0)": stationarity,
            "Primal Feasibility (Constraints Met)": primal_feasibility,
            "Dual Feasibility (μ ≥ 0)": dual_feasibility,
            "Complementary Slackness (μg=0)": complementary_slackness,
            "Stationarity Residual": np.linalg.norm(stationarity_grad),
            "Is_KKT_Point": all([stationarity, primal_feasibility, dual_feasibility, complementary_slackness])
        }
