import plotly.graph_objects as go
import numpy as np

def plot_generalized_eigenvalues(eigen_data):
    """
    Plots generalized eigenvalues on the complex plane, 
    separating finite targets from infinite (singular) points.
    """
    finite_reals = []
    finite_imags = []
    labels = []
    
    inf_count = 0
    indet_count = 0
    
    for d in eigen_data:
        if d['status'] == "Finite":
            finite_reals.append(np.real(d['eigenvalue']))
            finite_imags.append(np.imag(d['eigenvalue']))
            labels.append(f"λ = {d['eigenvalue']:.2f}")
        elif "Infinite" in d['status']:
            inf_count += 1
        else:
            indet_count += 1
            
    fig = go.Figure()
    
    if finite_reals:
        fig.add_trace(go.Scatter(
            x=finite_reals,
            y=finite_imags,
            mode='markers+text',
            marker=dict(size=12, color='darkorange', symbol='cross'),
            textposition="top center",
            name='Finite Generalized Eigenvalues'
        ))
        
    title = f"Generalized Eigenvalue Spectrum (Finite: {len(finite_reals)} | Inf: {inf_count} | Indeterminate: {indet_count})"
    
    fig.update_layout(
        title=title,
        xaxis_title="Real Axis",
        yaxis_title="Imaginary Axis",
        showlegend=True,
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black'),
        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black', scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig
