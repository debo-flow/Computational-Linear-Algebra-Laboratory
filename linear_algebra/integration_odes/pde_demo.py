import numpy as np

class PDEDemonstration:
    @staticmethod
    def heat_equation_1d(alpha, L, T, nx, nt, u0_func, u_left=0.0, u_right=0.0):
        """
        1D Heat Equation (u_t = alpha * u_xx) using Explicit Finite Difference (FTCS).
        """
        dx = L / (nx - 1)
        dt = T / nt
        r = alpha * dt / (dx**2)
        
        if r > 0.5:
            raise ValueError(f"Stability Condition Violated: r = {r:.4f} > 0.5. Decrease dt or increase dx.")
            
        x = np.linspace(0, L, nx)
        u = np.zeros((nt + 1, nx))
        
        # Initial Conditions
        u[0, :] = u0_func(x)
        u[0, 0] = u_left
        u[0, -1] = u_right
        
        # Time integration
        for n in range(nt):
            u[n+1, 1:-1] = u[n, 1:-1] + r * (u[n, 2:] - 2*u[n, 1:-1] + u[n, :-2])
            u[n+1, 0] = u_left
            u[n+1, -1] = u_right
            
        return x, np.linspace(0, T, nt+1), u
