import plotly.graph_objects as go
import numpy as np

def plot_optimization_path(func_eval, history_gd, history_nt=None):
    """
    Plots the convergence path of Optimization algorithms over a 2D scalar surface.
    func_eval: A callable f(x, y) returning a scalar.
    """
    history_gd = np.array(history_gd)
    
    # Determine bounds based on trajectory
    all_x = history_gd[:, 0]
    all_y = history_gd[:, 1]
    if history_nt is not None:
        history_nt = np.array(history_nt)
        all_x = np.concatenate([all_x, history_nt[:, 0]])
        all_y = np.concatenate([all_y, history_nt[:, 1]])
        
    x_min, x_max = all_x.min() - 1, all_x.max() + 1
    y_min, y_max = all_y.min() - 1, all_y.max() + 1
    
    x = np.linspace(x_min, x_max, 50)
    y = np.linspace(y_min, y_max, 50)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func_eval(np.array([X[i, j], Y[i, j]]))
            
    fig = go.Figure()
    
    # Surface contours
    fig.add_trace(go.Contour(
        x=x, y=y, z=Z, 
        colorscale='Viridis',
        opacity=0.5,
        name='Objective Surface'
    ))
    
    # GD Path
    fig.add_trace(go.Scatter(
        x=history_gd[:, 0], y=history_gd[:, 1],
        mode='lines+markers',
        marker=dict(size=6, color='red'),
        line=dict(color='red', width=2),
        name='Gradient Descent'
    ))
    
    # Newton Path
    if history_nt is not None:
        fig.add_trace(go.Scatter(
            x=history_nt[:, 0], y=history_nt[:, 1],
            mode='lines+markers',
            marker=dict(size=8, color='blue', symbol='x'),
            line=dict(color='blue', width=2, dash='dash'),
            name="Newton's Method"
        ))
        
    fig.update_layout(
        title="Optimization Trajectories",
        xaxis_title="x_1",
        yaxis_title="x_2",
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig
