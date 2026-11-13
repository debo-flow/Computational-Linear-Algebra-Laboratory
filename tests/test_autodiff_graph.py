import pytest
import numpy as np
from linear_algebra.calculus.autodiff import Dual
from linear_algebra.calculus.computational_graph.reverse_ad import Value
from linear_algebra.calculus.jvp_vjp import JVP_VJP_Engine

def test_reverse_ad_scalar():
    # f(x, y) = x^2 * y + exp(x)
    x = Value(2.0)
    y = Value(3.0)
    
    out = (x**2) * y + x.exp()
    out.backward()
    
    # df/dx = 2xy + exp(x) = 2(2)(3) + e^2 = 12 + 7.389 = 19.389056
    # df/dy = x^2 = 4
    expected_dx = 12.0 + np.exp(2.0)
    expected_dy = 4.0
    
    assert np.isclose(x.grad, expected_dx)
    assert np.isclose(y.grad, expected_dy)

def test_reverse_ad_division():
    # f(x, y) = x / y
    x = Value(4.0)
    y = Value(2.0)
    
    out = x / y
    out.backward()
    
    # df/dx = 1/y = 0.5
    # df/dy = -x/y^2 = -4/4 = -1.0
    assert np.isclose(x.grad, 0.5)
    assert np.isclose(y.grad, -1.0)

def test_jvp_forward_mode():
    # f(x, y) = [x^2 * y, x + y]
    # J = [[2xy, x^2], [1, 1]]
    # Evaluate at (x,y) = (2, 3) -> J = [[12, 4], [1, 1]]
    # Tangent v = [0.5, -0.5]
    # Jv = [[12, 4], [1, 1]] @ [0.5, -0.5]^T = [6 - 2, 0.5 - 0.5] = [4.0, 0.0]
    
    def f_dual(x, y):
        return [x**2 * y, x + y]
        
    y_val, jvp_val = JVP_VJP_Engine.jvp_forward(f_dual, [2.0, 3.0], [0.5, -0.5])
    
    assert np.allclose(jvp_val, [4.0, 0.0])

def test_vjp_reverse_mode():
    # Evaluate at (x,y) = (2, 3) -> J = [[12, 4], [1, 1]]
    # Cotangent u = [1.0, 1.0]
    # u^T J = [1, 1] @ [[12, 4], [1, 1]] = [13.0, 5.0]
    
    def f_val(x, y):
        return [x**2 * y, x + y]
        
    y_val, vjp_val = JVP_VJP_Engine.vjp_reverse(f_val, [2.0, 3.0], [1.0, 1.0])
    
    assert np.allclose(vjp_val, [13.0, 5.0])

def test_graph_zero_division():
    x = Value(4.0)
    y = Value(0.0)
    with pytest.raises(ZeroDivisionError):
        out = x / y
