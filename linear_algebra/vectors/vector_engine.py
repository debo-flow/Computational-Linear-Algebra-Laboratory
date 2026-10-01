import numpy as np
import sympy as sp

class VectorEngine:
    def __init__(self, data, is_symbolic=False):
        self.is_symbolic = is_symbolic
        self.vector = sp.Matrix(data) if is_symbolic else np.array(data, dtype=float).flatten()
        self.dim = len(self.vector)

    def magnitude(self):
        if self.is_symbolic:
            return sp.sqrt(self.vector.dot(self.vector))
        return np.linalg.norm(self.vector)

    def normalize(self):
        mag = self.magnitude()
        if float(mag) == 0:
            raise ValueError("Normalization undefined: Cannot normalize the zero vector.")
        if self.is_symbolic:
            return VectorEngine(self.vector / mag, True)
        return VectorEngine(self.vector / mag)

    @staticmethod
    def dot_product(u, v):
        if u.dim != v.dim:
            raise ValueError("Dot product undefined: Vectors must have the same dimension.")
        if u.is_symbolic or v.is_symbolic:
            return sp.Matrix(u.vector).dot(sp.Matrix(v.vector))
        return np.dot(u.vector, v.vector)

    @staticmethod
    def cross_product(u, v):
        if u.dim != 3 or v.dim != 3:
            raise ValueError("Cross product defined only for 3D vectors.")
        if u.is_symbolic or v.is_symbolic:
            u_sym, v_sym = sp.Matrix(u.vector), sp.Matrix(v.vector)
            return VectorEngine(u_sym.cross(v_sym), True)
        return VectorEngine(np.cross(u.vector, v.vector))
