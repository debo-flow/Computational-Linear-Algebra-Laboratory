import plotly.express as px

def plot_matrix_heatmap(A):
    fig = px.imshow(
        A.matrix, 
        text_auto=True, 
        aspect="auto",
        title="Matrix Heatmap (Structure & Magnitude)",
        color_continuous_scale="RdBu_r"
    )
    return fig
