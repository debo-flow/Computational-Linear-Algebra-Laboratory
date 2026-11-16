import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

class FiniteDifferencePDE:
    @staticmethod
    def solve_poisson_2d(mesh, f_rhs_func, bc_func):
        """
        Solves the 2D Poisson Equation: \nabla^2 u = f(x,y)
        Uses Sparse Matrix formulation with Dirichlet Boundary Conditions.
        """
        nx, ny = mesh.nx, mesh.ny
        dx, dy = mesh.dx, mesh.dy
        N = nx * ny
        
        # Build Sparse Matrix using DOK format for efficient construction
        A = sps.dok_matrix((N, N), dtype=float)
        b = np.zeros(N)
        
        dx2, dy2 = dx**2, dy**2
        
        for j in range(ny):
            for i in range(nx):
                k = j * nx + i
                x_val, y_val = mesh.x[i], mesh.y[j]
                
                # Boundary Conditions
                if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                    A[k, k] = 1.0
                    b[k] = bc_func(x_val, y_val)
                else:
                    # Central Difference Stencil
                    A[k, k] = -2.0 / dx2 - 2.0 / dy2
                    A[k, k - 1] = 1.0 / dx2
                    A[k, k + 1] = 1.0 / dx2
                    A[k, k - nx] = 1.0 / dy2
                    A[k, k + nx] = 1.0 / dy2
                    b[k] = f_rhs_func(x_val, y_val)
                    
        A_csr = A.tocsr()
        u = spla.spsolve(A_csr, b)
        return u.reshape((ny, nx))

    @staticmethod
    def solve_heat_1d_crank_nicolson(mesh, T_max, nt, alpha, u0_func, bc_left, bc_right):
        """
        Solves 1D Heat Eq: u_t = \alpha u_{xx} using Unconditionally Stable Crank-Nicolson.
        (I - r/2 D2) u^{n+1} = (I + r/2 D2) u^n
        """
        dt = T_max / nt
        r = alpha * dt / (mesh.dx ** 2)
        
        nx = mesh.nx
        u = np.zeros((nt + 1, nx))
        u[0, :] = u0_func(mesh.x)
        u[:, 0] = bc_left
        u[:, -1] = bc_right
        
        # Matrix Setup
        main_diag_A = (1 + r) * np.ones(nx - 2)
        off_diag_A = (-r / 2) * np.ones(nx - 3)
        A = sps.diags([off_diag_A, main_diag_A, off_diag_A], [-1, 0, 1], format='csr')
        
        main_diag_B = (1 - r) * np.ones(nx - 2)
        off_diag_B = (r / 2) * np.ones(nx - 3)
        B = sps.diags([off_diag_B, main_diag_B, off_diag_B], [-1, 0, 1], format='csr')
        
        for n in range(nt):
            rhs = B.dot(u[n, 1:-1])
            # Apply Boundary Conditions
            rhs[0] += (r / 2) * (u[n, 0] + u[n+1, 0])
            rhs[-1] += (r / 2) * (u[n, -1] + u[n+1, -1])
            
            u[n+1, 1:-1] = spla.spsolve(A, rhs)
            
        return u

    @staticmethod
    def solve_wave_1d(mesh, T_max, nt, c, u0_func, v0_func):
        """
        Solves 1D Wave Eq: u_tt = c^2 u_{xx} using Leapfrog Explicit Central Difference.
        Must satisfy CFL Condition: c * dt / dx <= 1
        """
        dt = T_max / nt
        CFL = c * dt / mesh.dx
        
        if CFL > 1.0:
            raise ValueError(f"Numerical Instability: CFL number {CFL:.4f} > 1.0. Decrease time step or increase mesh resolution.")
            
        nx = mesh.nx
        u = np.zeros((nt + 1, nx))
        
        # Initial position
        u[0, :] = u0_func(mesh.x)
        # Initial velocity boundary condition formulation for step 1
        v0 = v0_func(mesh.x)
        
        # Step 1: Use ghost nodes and Taylor expansion for first time step
        for i in range(1, nx - 1):
            u[1, i] = u[0, i] + dt * v0[i] + 0.5 * CFL**2 * (u[0, i+1] - 2*u[0, i] + u[0, i-1])
        u[1, 0] = 0; u[1, -1] = 0 # Fixed boundaries
        
        # Step n: Standard leapfrog iteration
        for n in range(1, nt):
            u[n+1, 1:-1] = 2*u[n, 1:-1] - u[n-1, 1:-1] + CFL**2 * (u[n, 2:] - 2*u[n, 1:-1] + u[n, :-2])
            u[n+1, 0] = 0; u[n+1, -1] = 0
            
        return u

    @staticmethod
    def solve_burgers_1d(mesh, T_max, nt, nu, u0_func):
        """
        Solves 1D Nonlinear Burgers' Equation: u_t + u u_x = \nu u_{xx}
        Uses explicit FTCS (Forward Time Central Space).
        """
        dt = T_max / nt
        nx, dx = mesh.nx, mesh.dx
        u = np.zeros((nt + 1, nx))
        u[0, :] = u0_func(mesh.x)
        
        # Stability Check
        stability_limit = nu * dt / (dx**2)
        if stability_limit > 0.5:
            raise ValueError(f"Diffusion stability violated: {stability_limit:.4f} > 0.5")
            
        for n in range(nt):
            un = u[n, :]
            # Non-linear Advection Term + Linear Diffusion Term
            u[n+1, 1:-1] = un[1:-1] - un[1:-1] * (dt / (2*dx)) * (un[2:] - un[:-2]) \
                           + (nu * dt / dx**2) * (un[2:] - 2*un[1:-1] + un[:-2])
            
            # Periodic boundaries
            u[n+1, 0] = un[0] - un[0] * (dt / (2*dx)) * (un[1] - un[-2]) + (nu * dt / dx**2) * (un[1] - 2*un[0] + un[-2])
            u[n+1, -1] = u[n+1, 0]
            
        return u
