import numpy as np

class FiniteVolumePDE:
    @staticmethod
    def solve_advection_1d(mesh, T_max, nt, velocity, u0_func):
        """
        1D Finite Volume Method for Advection Eq: u_t + v u_x = 0
        Uses Cell-Centered Upwind Flux formulation.
        """
        dt = T_max / nt
        dx = mesh.dx
        CFL = velocity * dt / dx
        
        if abs(CFL) > 1.0:
            raise ValueError(f"FVM Stability limit exceeded. CFL {CFL:.4f} > 1.0")
            
        u = np.zeros((nt + 1, mesh.nx))
        u[0, :] = u0_func(mesh.x)
        
        for n in range(nt):
            # Upwind numerical flux
            F = np.zeros(mesh.nx + 1)
            for i in range(1, mesh.nx):
                F[i] = velocity * u[n, i-1] if velocity > 0 else velocity * u[n, i]
            
            # Periodic boundaries for flux
            F[0] = velocity * u[n, -1] if velocity > 0 else velocity * u[n, 0]
            F[-1] = F[0]
            
            # Update Cell Averages
            for i in range(mesh.nx):
                u[n+1, i] = u[n, i] - (dt / dx) * (F[i+1] - F[i])
                
        return u
