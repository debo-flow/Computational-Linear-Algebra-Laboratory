import plotly.graph_objects as go
import numpy as np

def plot_numerical_integration(f, a, b, nodes, values, method_name):
    """Plots the function and the quadrature/Riemann approximations."""
    x_fine = np.linspace(a, b, 500)
    y_fine = f(x_fine)
    
    fig = go.Figure()
    # True function
    fig.add_trace(go.Scatter(x=x_fine, y=y_fine, mode='lines', name='f(x)', line=dict(color='blue')))
    
    # Quadrature points/bars
    if 'Gauss' in method_name:
        fig.add_trace(go.Scatter(x=nodes, y=values, mode='markers', name='Gauss Nodes', marker=dict(size=10, color='red')))
    else:
        # Fill polygons for Riemann/Trapezoid
        fig.add_trace(go.Scatter(x=nodes, y=values, mode='lines+markers', name=method_name, fill='tozeroy', line=dict(color='orange', dash='dash')))
        
    fig.update_layout(title=f"Integration: {method_name}", xaxis_title="x", yaxis_title="f(x)", margin=dict(l=0, r=0, b=0, t=40))
    return fig

def plot_ode_trajectories(t, y_vals, title="ODE Solution Trajectories"):
    """Plots time series for ODE solutions."""
    fig = go.Figure()
    num_vars = y_vals.shape[1]
    
    for i in range(num_vars):
        fig.add_trace(go.Scatter(x=t, y=y_vals[:, i], mode='lines+markers', marker=dict(size=4), name=f'y_{i}(t)'))
        
    fig.update_layout(title=title, xaxis_title="Time (t)", yaxis_title="State (y)", margin=dict(l=0, r=0, b=0, t=40))
    return fig

def plot_pde_heatmap(x, t, u, title="1D Heat Equation Evolution"):
    """Plots the spatio-temporal evolution of a 1D PDE."""
    fig = go.Figure(data=go.Heatmap(
        z=u, x=x, y=t, colorscale='Inferno', colorbar=dict(title='Heat u(x,t)')
    ))
    fig.update_layout(title=title, xaxis_title="Space (x)", yaxis_title="Time (t)", margin=dict(l=0, r=0, b=0, t=40))
    return fig
