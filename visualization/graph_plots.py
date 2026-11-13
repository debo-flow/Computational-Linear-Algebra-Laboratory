import plotly.graph_objects as go

def plot_computational_graph(root_node):
    """
    Renders a directed Computational Graph built from Reverse-Mode Value nodes using Plotly.
    Builds layered coordinates topologically.
    """
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root_node)
    
    # Simple Layout algorithm (Left to Right)
    node_to_id = {n: i for i, n in enumerate(nodes)}
    
    # Trace depth
    depths = {}
    def calc_depth(v):
        if v in depths: return depths[v]
        if not v._prev: 
            depths[v] = 0
            return 0
        d = 1 + max(calc_depth(c) for c in v._prev)
        depths[v] = d
        return d
        
    for n in nodes: calc_depth(n)
    max_d = max(depths.values()) if depths else 1
    
    # Assign coordinates
    layer_counts = {}
    positions = {}
    for n in nodes:
        d = depths[n]
        c = layer_counts.get(d, 0)
        positions[n] = (d, c)
        layer_counts[d] = c + 1
        
    node_x = []
    node_y = []
    node_text = []
    
    for n in nodes:
        x_c, y_c = positions[n]
        # Center Y structurally
        y_centered = y_c - layer_counts[x_c]/2.0
        node_x.append(x_c * 2)
        node_y.append(y_centered * 1.5)
        
        label = n.label if n.label else n._op
        txt = f"{label}<br>v: {n.data:.3g}<br>∇: {n.grad:.3g}"
        node_text.append(txt)
        
    edge_x, edge_y = [], []
    for src, dst in edges:
        x0, y0 = positions[src][0]*2, (positions[src][1] - layer_counts[positions[src][0]]/2.0)*1.5
        x1, y1 = positions[dst][0]*2, (positions[dst][1] - layer_counts[positions[dst][0]]/2.0)*1.5
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='gray', width=1.5), hoverinfo='none'))
    
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        marker=dict(size=35, color='lightblue', line=dict(width=2, color='darkblue')),
        text=node_text, textposition="bottom center", hoverinfo='text'
    ))
    
    fig.update_layout(
        title="Directed Computational Graph (Reverse-Mode AD)",
        showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=20, r=20, b=20, t=40)
    )
    return fig
