import plotly.graph_objects as go
import numpy as np

def plot_ode_timeseries(t_vals, X_vals):
    """Plots individual state components x_i(t) over time."""
    fig = go.Figure()
    num_states = X_vals.shape[1]
    
    for i in range(num_states):
        fig.add_trace(go.Scatter(
            x=t_vals, 
            y=np.real(X_vals[:, i]), 
            mode='lines', 
            name=f'x_{i+1}(t)'
        ))
        
    fig.update_layout(
        title="Time Series Trajectories",
        xaxis_title="Time (t)",
        yaxis_title="State Values",
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig

def plot_ode_phase_space(X_vals):
    """Plots 2D Phase Portrait (x_1 vs x_2)."""
    fig = go.Figure()
    
    if X_vals.shape[1] < 2:
        return fig
        
    fig.add_trace(go.Scatter(
        x=np.real(X_vals[:, 0]), 
        y=np.real(X_vals[:, 1]), 
        mode='lines+markers',
        marker=dict(size=4),
        name='Trajectory'
    ))
    
    # Mark origin and start point
    fig.add_trace(go.Scatter(x=[np.real(X_vals[0, 0])], y=[np.real(X_vals[0, 1])], mode='markers', marker=dict(color='green', size=10), name='Start (t=0)'))
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker=dict(color='red', symbol='cross', size=10), name='Origin'))
    
    fig.update_layout(
        title="2D Phase Space: x_1(t) vs x_2(t)",
        xaxis_title="x_1",
        yaxis_title="x_2",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig

def plot_eigenvalue_mapping(orig_w, mapped_w, title="Eigenvalue Spectral Mapping"):
    """Plots z -> f(z) in the complex plane."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=np.real(orig_w), y=np.imag(orig_w),
        mode='markers+text', text=["λ" for _ in orig_w],
        marker=dict(size=10, color='blue', symbol='circle'), name='Original λ'
    ))
    
    fig.add_trace(go.Scatter(
        x=np.real(mapped_w), y=np.imag(mapped_w),
        mode='markers+text', text=["f(λ)" for _ in mapped_w],
        marker=dict(size=10, color='red', symbol='x'), name='Mapped f(λ)'
    ))
    
    fig.update_layout(
        title=title, xaxis_title="Real", yaxis_title="Imaginary",
        xaxis=dict(zeroline=True), yaxis=dict(zeroline=True, scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig
