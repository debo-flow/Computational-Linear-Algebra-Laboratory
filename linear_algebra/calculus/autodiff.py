import math
import numpy as np

class Dual:
    """
    Educational Forward-Mode Automatic Differentiation Engine using Dual Numbers.
    Represents numbers of the form a + b*epsilon where epsilon^2 = 0.
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
            # a^b = exp(b * log(a))
            if self.real <= 0:
                raise ValueError("Base must be positive for Dual^Dual.")
            real_part = self.real ** power.real
            dual_part = real_part * (power.dual * math.log(self.real) + power.real * self.dual / self.real)
            return Dual(real_part, dual_part)
        real_part = self.real ** power
        dual_part = power * (self.real ** (power - 1)) * self.dual if (self.real != 0 or power > 0) else 0.0
        return Dual(real_part, dual_part)

    def __neg__(self):
        return Dual(-self.real, -self.dual)

    def __abs__(self):
        return Dual(abs(self.real), self.dual if self.real >= 0 else -self.dual)
        
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

    def sqrt(self):
        if self.real < 0:
            raise ValueError("Sqrt domain error.")
        return Dual(math.sqrt(self.real), 0.5 * self.dual / math.sqrt(self.real) if self.real > 0 else 0.0)

class AutoDiffEngine:
    @staticmethod
    def gradient(func, x_val):
        """Computes gradient of a scalar function via Forward-Mode AD."""
        n = len(x_val)
        grad = np.zeros(n)
        for i in range(n):
            x_dual = [Dual(x_val[j], 1.0 if i == j else 0.0) for j in range(n)]
            y_dual = func(*x_dual)
            grad[i] = y_dual.dual
        return grad

    @staticmethod
    def jacobian(func_vec, x_val, output_dim):
        """Computes Jacobian matrix via Forward-Mode AD."""
        n = len(x_val)
        jac = np.zeros((output_dim, n))
        for i in range(n):
            x_dual = [Dual(x_val[j], 1.0 if i == j else 0.0) for j in range(n)]
            y_dual_vec = func_vec(*x_dual)
            for k in range(output_dim):
                jac[k, i] = y_dual_vec[k].dual
        return jac
