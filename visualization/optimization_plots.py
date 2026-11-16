import plotly.graph_objects as go
import numpy as np

def plot_2d_optimization_trajectory(func, histories_dict, bounds=((-5, 5), (-5, 5))):
    """
    Plots the contour of the objective function alongside optimization trajectories.
    histories_dict: {"Algorithm Name": np.array(history_points)}
    """
    x_range, y_range = bounds
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            val, _, _ = func(np.array([X[i, j], Y[i, j]]))
            Z[i, j] = val
            
    fig = go.Figure()
    
    # Contour Plot (log scale for high-curvature functions like Rosenbrock)
    fig.add_trace(go.Contour(
        x=x, y=y, z=np.log1p(Z - np.min(Z)), 
        colorscale='Viridis',
        opacity=0.6,
        name='Log(Objective)',
        showscale=False
    ))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for idx, (name, hist) in enumerate(histories_dict.items()):
        hist = np.array(hist)
        fig.add_trace(go.Scatter(
            x=hist[:, 0], y=hist[:, 1],
            mode='lines+markers',
            marker=dict(size=6, symbol='circle'),
            line=dict(width=2, color=colors[idx % len(colors)]),
            name=name
        ))
        
        # Mark Start and End
        fig.add_trace(go.Scatter(
            x=[hist[0, 0]], y=[hist[0, 1]],
            mode='markers', marker=dict(size=10, color=colors[idx % len(colors)], symbol='square'),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[hist[-1, 0]], y=[hist[-1, 1]],
            mode='markers', marker=dict(size=12, color=colors[idx % len(colors)], symbol='star'),
            showlegend=False
        ))

    fig.update_layout(
        title="Optimization Landscape & Trajectories",
        xaxis_title="x_1",
        yaxis_title="x_2",
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig
