import plotly.graph_objects as go
import numpy as np

def plot_pde_surface_2d(mesh, u, title="2D PDE Solution"):
    """Plots 3D Surface for 2D Elliptic PDEs (Poisson/Laplace)."""
    fig = go.Figure(data=[go.Surface(z=u, x=mesh.x, y=mesh.y, colorscale='Viridis')])
    fig.update_layout(
        title=title, scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="u(x,y)"),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig

def plot_pde_evolution_1d(mesh, t_array, u_matrix, title="1D Time Evolution"):
    """Plots temporal snapshots of a 1D PDE."""
    fig = go.Figure()
    
    # Plot Initial, Middle, and Final States
    indices = [0, len(t_array)//2, len(t_array)-1]
    labels = ["t = 0 (Initial)", f"t = {t_array[indices[1]]:.2f}", f"t = {t_array[-1]:.2f} (Final)"]
    colors = ['blue', 'orange', 'red']
    
    for idx, label, color in zip(indices, labels, colors):
        fig.add_trace(go.Scatter(x=mesh.x, y=u_matrix[idx, :], mode='lines', name=label, line=dict(color=color)))
        
    fig.update_layout(title=title, xaxis_title="Domain (x)", yaxis_title="Amplitude u(x,t)", margin=dict(l=0, r=0, b=0, t=30))
    return fig
