import plotly.graph_objects as go
import numpy as np

def plot_3d_vectors(vectors, labels=None):
    """Plots a list of 3D vectors from origin."""
    fig = go.Figure()
    
    for i, vec in enumerate(vectors):
        v = vec.vector
        label = labels[i] if labels else f"v{i+1}"
        fig.add_trace(go.Scatter3d(
            x=[0, v[0]], y=[0, v[1]], z=[0, v[2]],
            mode='lines+markers+text',
            text=["", label],
            textposition="top center",
            line=dict(width=5),
            marker=dict(size=[0, 5], symbol='arrow-up'),
            name=label
        ))
        
    fig.update_layout(
        title="3D Vector Visualization",
        scene=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Z Axis'),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig
