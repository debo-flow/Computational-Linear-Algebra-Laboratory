import plotly.graph_objects as go
import numpy as np

def plot_2d_transformation(A_mat, grid_density=15):
    """Visualizes the effect of a 2D transformation on a unit circle and grid."""
    fig = go.Figure()

    # Original Unit Circle
    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.vstack((np.cos(theta), np.sin(theta)))
    
    # Transformed Circle
    transformed_circle = A_mat @ circle
    
    fig.add_trace(go.Scatter(x=circle[0], y=circle[1], mode='lines', 
                             name='Original Unit Circle', line=dict(dash='dash', color='blue')))
    fig.add_trace(go.Scatter(x=transformed_circle[0], y=transformed_circle[1], mode='lines', 
                             name='Transformed Image', line=dict(color='red', width=3)))

    # Basis vectors i, j
    bases = np.array([[1, 0], [0, 1]])
    t_bases = A_mat @ bases
    
    fig.add_trace(go.Scatter(x=[0, bases[0,0]], y=[0, bases[1,0]], mode='lines+markers', name='i-hat (orig)', line=dict(color='lightblue')))
    fig.add_trace(go.Scatter(x=[0, bases[0,1]], y=[0, bases[1,1]], mode='lines+markers', name='j-hat (orig)', line=dict(color='lightgreen')))
    
    fig.add_trace(go.Scatter(x=[0, t_bases[0,0]], y=[0, t_bases[1,0]], mode='lines+markers', name='T(i-hat)', line=dict(color='darkblue', width=4)))
    fig.add_trace(go.Scatter(x=[0, t_bases[0,1]], y=[0, t_bases[1,1]], mode='lines+markers', name='T(j-hat)', line=dict(color='darkgreen', width=4)))

    fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1), 
                      title="2D Geometric Transformation Mapping",
                      margin=dict(l=0, r=0, b=0, t=30))
    return fig
