import math
import numpy as np

class Dual:
    """
    Educational Forward-Mode Automatic Differentiation Engine using Dual Numbers.
    Represents numbers of the form a + b*epsilon where epsilon^2 = 0.
    'real' is the value, 'dual' is the derivative.
    """
    def __init__(self, real, dual=0.0):
        self.real = float(real)
        self.dual = float(dual)
        
    def __add__(self, other):
        other = other if isinstance(other, Dual) else Dual(other)
        return Dual(self.real + other.real, self.dual + other.dual)
        
    def __radd__(self, other):
        return self.__add__(other)
        
    def __sub__(self, other):
        other = other if isinstance(other, Dual) else Dual(other)
        return Dual(self.real - other.real, self.dual - other.dual)
        
    def __rsub__(self, other):
        other = other if isinstance(other, Dual) else Dual(other)
        return Dual(other.real - self.real, other.dual - self.dual)
        
    def __mul__(self, other):
        other = other if isinstance(other, Dual) else Dual(other)
        return Dual(self.real * other.real, self.real * other.dual + self.dual * other.real)
        
    def __rmul__(self, other):
        return self.__mul__(other)
        
    def __truediv__(self, other):
        other = other if isinstance(other, Dual) else Dual(other)
        if other.real == 0:
            raise ZeroDivisionError("Division by zero in Dual number.")
        real_part = self.real / other.real
        dual_part = (self.dual * other.real - self.real * other.dual) / (other.real ** 2)
        return Dual(real_part, dual_part)
        
    def __rtruediv__(self, other):
        other = other if isinstance(other, Dual) else Dual(other)
        return other.__truediv__(self)
        
    def __pow__(self, power):
        if isinstance(power, Dual):
            raise NotImplementedError("Dual to the power of Dual not implemented in educational engine.")
        real_part = self.real ** power
        dual_part = power * (self.real ** (power - 1)) * self.dual if self.real != 0 or power > 0 else 0
        return Dual(real_part, dual_part)
        
    def sin(self):
        return Dual(math.sin(self.real), math.cos(self.real) * self.dual)
        
    def cos(self):
        return Dual(math.cos(self.real), -math.sin(self.real) * self.dual)
        
    def exp(self):
        return Dual(math.exp(self.real), math.exp(self.real) * self.dual)
        
    def log(self):
        if self.real <= 0:
            raise ValueError("Logarithm domain error.")
        return Dual(math.log(self.real), self.dual / self.real)

class AutoDiffEngine:
    @staticmethod
    def gradient(func, x_val):
        """Computes the gradient of a scalar function using Forward-Mode AD."""
        n = len(x_val)
        grad = np.zeros(n)
        for i in range(n):
            # Seed the i-th direction with dual=1, rest dual=0
            x_dual = [Dual(x_val[j], 1.0 if i == j else 0.0) for j in range(n)]
            y_dual = func(*x_dual)
            grad[i] = y_dual.dual
        return grad
