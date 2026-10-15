import numpy as np
import sympy as sp
from linear_algebra.matrix.matrix_engine import MatrixEngine

class GeometricTransformations:
    @staticmethod
    def rotation_2d(theta, is_degrees=False, is_symbolic=False):
        """Generates a 2D rotation matrix R(theta)."""
        if is_symbolic:
            angle = theta * sp.pi / 180 if is_degrees else theta
            mat = sp.Matrix([[sp.cos(angle), -sp.sin(angle)],
                             [sp.sin(angle),  sp.cos(angle)]])
            return MatrixEngine(mat, True)
        else:
            angle = np.radians(theta) if is_degrees else float(theta)
            c, s = np.cos(angle), np.sin(angle)
            return MatrixEngine([[c, -s], [s, c]])

    @staticmethod
    def scaling(scales, is_symbolic=False):
        """Generates an n-dimensional scaling matrix."""
        if is_symbolic:
            return MatrixEngine(sp.diag(*scales), True)
        return MatrixEngine(np.diag(scales))

    @staticmethod
    def reflection_2d(axis='x', is_symbolic=False):
        """Generates a 2D reflection matrix."""
        if axis == 'x':
            mat = [[1, 0], [0, -1]]
        elif axis == 'y':
            mat = [[-1, 0], [0, 1]]
        elif axis == 'y=x':
            mat = [[0, 1], [1, 0]]
        else:
            raise ValueError("Unsupported 2D reflection axis. Use 'x', 'y', or 'y=x'.")
        return MatrixEngine(sp.Matrix(mat) if is_symbolic else np.array(mat, dtype=float), is_symbolic)

    @staticmethod
    def shear_2d(k, axis='x', is_symbolic=False):
        """Generates a 2D shear matrix."""
        mat = [[1, k] if axis == 'x' else [0, 1],
               [k, 1] if axis == 'y' else [0, 1]] # corrected mapping below
        
        if axis == 'x':
            mat = [[1, k], [0, 1]]
        elif axis == 'y':
            mat = [[1, 0], [k, 1]]
        else:
            raise ValueError("Shear axis must be 'x' or 'y'.")
            
        return MatrixEngine(sp.Matrix(mat) if is_symbolic else np.array(mat, dtype=float), is_symbolic)

    @staticmethod
    def rotation_3d(theta, axis='z', is_degrees=False, is_symbolic=False):
        """Generates a 3D rotation matrix around a standard axis."""
        if is_symbolic:
            angle = theta * sp.pi / 180 if is_degrees else theta
            c, s = sp.cos(angle), sp.sin(angle)
        else:
            angle = np.radians(theta) if is_degrees else float(theta)
            c, s = np.cos(angle), np.sin(angle)

        if axis == 'z':
            mat = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        elif axis == 'y':
            mat = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        elif axis == 'x':
            mat = [[1, 0, 0], [0, c, -s], [0, s, c]]
        else:
            raise ValueError("3D Rotation axis must be 'x', 'y', or 'z'.")

        return MatrixEngine(sp.Matrix(mat) if is_symbolic else np.array(mat, dtype=float), is_symbolic)
