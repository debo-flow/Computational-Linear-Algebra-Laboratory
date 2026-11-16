import numpy as np

class Mesh1D:
    def __init__(self, L, nx):
        self.L = L
        self.nx = nx
        self.x = np.linspace(0, L, nx)
        self.dx = self.x[1] - self.x[0]

class Mesh2D:
    def __init__(self, Lx, Ly, nx, ny):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.X, self.Y = np.meshgrid(self.x, self.y)
