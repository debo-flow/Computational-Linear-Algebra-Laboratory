import plotly.graph_objects as go
import numpy as np

def plot_spy(mat, title="Sparse Matrix Pattern (Spy Plot)"):
    """
    Creates a Spy plot (scatter of non-zero elements).
    Limits plotting points to prevent browser crashing on massive matrices.
    """
    coo = mat.tocoo()
    
    # Browser safety limit for Plotly scatter plots
    MAX_POINTS = 50_000
    if coo.nnz > MAX_POINTS:
        # Sample points uniformly if too dense for browser render
        idx = np.random.choice(coo.nnz, MAX_POINTS, replace=False)
        rows = coo.row[idx]
        cols = coo.col[idx]
        warning = f" (Sampled {MAX_POINTS} / {coo.nnz} points for browser safety)"
    else:
        rows = coo.row
        cols = coo.col
        warning = ""
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cols, y=rows,
        mode='markers',
        marker=dict(size=2, color='darkblue', symbol='square'),
        hoverinfo='none'
    ))
    
    # Invert Y axis so (0,0) is top-left, matching standard matrix notation
    fig.update_layout(
        title=title + warning,
        xaxis=dict(title="Columns", range=[-0.5, mat.shape[1]-0.5]),
        yaxis=dict(title="Rows", range=[mat.shape[0]-0.5, -0.5], scaleanchor="x", scaleratio=1),
        plot_bgcolor='white',
        width=600, height=600,
        margin=dict(l=20, r=20, b=20, t=40)
    )
    return fig

def plot_convergence(history, method_name="Solver"):
    """Plots iterative solver convergence history."""
    fig = go.Figure()
    # The callback in scipy might return relative residuals depending on the method
    # It might also return None on exact 0, handled safely
    vals = [float(v) for v in history if v is not None]
    
    fig.add_trace(go.Scatter(
        x=np.arange(1, len(vals)+1), y=vals,
        mode='lines+markers', line=dict(color='red')
    ))
    
    fig.update_layout(
        title=f"{method_name} Convergence History",
        xaxis_title="Iteration",
        yaxis_title="Residual Norm",
        yaxis_type="log",
        margin=dict(l=20, r=20, b=20, t=40)
    )
    return fig
