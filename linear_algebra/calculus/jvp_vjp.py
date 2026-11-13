import numpy as np
from .autodiff import Dual
from .computational_graph.reverse_ad import Value

class JVP_VJP_Engine:
    @staticmethod
    def jvp_forward(func_vec, x_val, v_val):
        """
        Computes Jacobian-Vector Product (JVP) using Forward-Mode Dual Numbers.
        J(x)v is exactly the dual component of f(x + epsilon * v).
        """
        n = len(x_val)
        # x_dual_i = x_i + epsilon * v_i
        x_dual = [Dual(x_val[i], v_val[i]) for i in range(n)]
        y_dual_vec = func_vec(*x_dual)
        
        y_val = [y.real for y in y_dual_vec]
        jvp_val = [y.dual for y in y_dual_vec]
        
        return np.array(y_val), np.array(jvp_val)

    @staticmethod
    def vjp_reverse(func_vec, x_val, u_val):
        """
        Computes Vector-Jacobian Product (VJP) using Reverse-Mode Computational Graph.
        u^T J(x) is exactly x.grad after seeding y.grad = u and backpropagating.
        """
        n = len(x_val)
        m = len(u_val)
        
        x_nodes = [Value(x_val[i], label=f'x_{i}') for i in range(n)]
        y_nodes = func_vec(*x_nodes)
        
        y_val = [y.data for y in y_nodes]
        
        # Seed gradients and backpropagate
        for i in range(m):
            y_nodes[i].grad = u_val[i]
            y_nodes[i].backward()
            
        vjp_val = [x.grad for x in x_nodes]
        return np.array(y_val), np.array(vjp_val)
