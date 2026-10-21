import plotly.graph_objects as go
import numpy as np

def plot_2d_quadratic_form(A_mat):
    """
    Plots the 3D surface and 2D contour of q(x,y) = [x, y] A [x, y]^T.
    Assumes A_mat is a 2x2 real symmetric matrix.
    """
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate Z = A_00*X^2 + (A_01 + A_10)*XY + A_11*Y^2
    Z = A_mat[0, 0] * X**2 + (A_mat[0, 1] + A_mat[1, 0]) * X * Y + A_mat[1, 1] * Y**2

    fig = go.Figure()

    # 3D Surface Plot
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z, 
        colorscale='Viridis', 
        opacity=0.8,
        name='q(x,y) Surface'
    ))

    # Add a zero-plane to clearly show indefiniteness boundaries
    fig.add_trace(go.Surface(
        x=X, y=Y, z=np.zeros_like(Z),
        colorscale='RdBu', showscale=False, opacity=0.3,
        name='Zero Plane'
    ))

    fig.update_layout(
        title="Quadratic Form Geometry q(x,y)",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='q(x, y)'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig
