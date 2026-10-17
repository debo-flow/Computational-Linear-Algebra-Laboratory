import plotly.graph_objects as go
import numpy as np

def plot_singular_values(S, log_scale=False):
    """Plots the singular value spectrum as a bar chart."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"σ_{i+1}" for i in range(len(S))],
        y=S,
        marker_color='royalblue'
    ))
    
    fig.update_layout(
        title="Singular Value Spectrum",
        xaxis_title="Index",
        yaxis_title="Magnitude",
        yaxis_type="log" if log_scale else "linear",
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig

def plot_energy_retention(S):
    """Plots cumulative squared singular value energy."""
    energy = S**2
    cum_energy = np.cumsum(energy)
    total = cum_energy[-1] if len(cum_energy) > 0 else 1.0
    cum_ratio = cum_energy / total
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[f"k={i+1}" for i in range(len(S))],
        y=cum_ratio * 100,
        mode='lines+markers',
        marker=dict(size=8, color='firebrick')
    ))
    
    fig.update_layout(
        title="Cumulative Retained Energy (%)",
        xaxis_title="Truncation Rank k",
        yaxis_title="Energy %",
        yaxis=dict(range=[0, 105]),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig
