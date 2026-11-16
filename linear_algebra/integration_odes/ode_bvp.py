import numpy as np
import scipy.integrate as spi

class ODESolverBVP:
    @staticmethod
    def solve_2nd_order_bvp(f_ode, bc_func, x_nodes, y_guess):
        """
        Solves a 2nd-order Boundary Value Problem using Collocation.
        f_ode(x, y) returns [y', y'']
        bc_func(ya, yb) returns residuals of boundary conditions.
        """
        sol = spi.solve_bvp(f_ode, bc_func, x_nodes, y_guess)
        if not sol.success:
            raise RuntimeError(f"BVP Solver failed: {sol.message}")
        return sol.x, sol.y, sol.rms_residuals
