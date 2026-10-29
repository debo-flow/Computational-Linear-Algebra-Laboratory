import plotly.graph_objects as go
import numpy as np

def plot_eigenvalue_spectrum(orig_eigvals, pert_eigvals=None):
    """Plots eigenvalues in the complex plane."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=np.real(orig_eigvals),
        y=np.imag(orig_eigvals),
        mode='markers',
        marker=dict(size=12, color='blue', symbol='circle'),
        name='Original Eigenvalues'
    ))
    
    if pert_eigvals is not None:
        fig.add_trace(go.Scatter(
            x=np.real(pert_eigvals),
            y=np.imag(pert_eigvals),
            mode='markers',
            marker=dict(size=10, color='red', symbol='x'),
            name='Perturbed Eigenvalues'
        ))
        
    # Formatting
    fig.update_layout(
        title="Eigenvalue Spectrum (Complex Plane)",
        xaxis_title="Real Axis",
        yaxis_title="Imaginary Axis",
        showlegend=True,
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black'),
        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black', scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig

def plot_convergence_history(history, metric='residual'):
    """Plots residual or off-diagonal norm history."""
    iters = [h['iter'] for h in history]
    
    if metric == 'residual':
        vals = [h['residual'] for h in history]
        title = "Residual Convergence History"
        y_title = "||Ax - λx||"
    else:
        vals = [h['lower_off_diag_norm'] for h in history]
        title = "QR Off-Diagonal Norm Convergence"
        y_title = "||Tril(A_k)|| (Frobenius)"
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=iters, y=vals, mode='lines+markers', line=dict(color='green')))
    fig.update_layout(
        title=title, xaxis_title="Iteration", yaxis_title=y_title,
        yaxis_type="log", margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig
