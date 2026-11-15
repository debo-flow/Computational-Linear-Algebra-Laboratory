import plotly.graph_objects as go

def plot_residual_history(histories_dict):
    """
    Plots the convergence history (relative residual norm vs iteration) for multiple solvers.
    histories_dict: dict of { "Solver Name": [res_0, res_1, ...] }
    """
    fig = go.Figure()
    
    for name, hist in histories_dict.items():
        # Prevent log(0) warnings natively by clipping
        y_vals = [max(h, 1e-16) for h in hist]
        fig.add_trace(go.Scatter(
            x=list(range(len(y_vals))),
            y=y_vals,
            mode='lines+markers',
            name=name,
            marker=dict(size=4)
        ))
        
    fig.update_layout(
        title="Iterative Solver Convergence History",
        xaxis_title="Iteration Number k",
        yaxis_title="Relative Residual ||b - Ax_k|| / ||b||",
        yaxis_type="log",
        showlegend=True,
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig
