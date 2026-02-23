import io

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="GNN Graph Explorer", page_icon="🕸️", layout="wide")

st.title("🕸️ GNN Graph Explorer")
st.caption(
    "Visualise graph structure and GCN message passing interactively — "
    "no training, no API key."
)

# ── Built-in datasets ─────────────────────────────────────────────────────────
BUILTIN = {
    "Karate Club": nx.karate_club_graph,
    "Les Misérables": nx.les_miserables_graph,
    "Florentine Families": nx.florentine_families_graph,
    "Barabási-Albert (n=60)": lambda: nx.barabasi_albert_graph(60, 3, seed=42),
    "Watts-Strogatz (n=60)": lambda: nx.watts_strogatz_graph(60, 4, 0.3, seed=42),
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Graph")
    graph_choice = st.selectbox("Dataset", list(BUILTIN.keys()) + ["Upload CSV"])
    uploaded_csv = None
    if graph_choice == "Upload CSV":
        uploaded_csv = st.file_uploader("CSV edge list (source, target)", type="csv")

    st.divider()
    st.header("GCN settings")
    n_layers = st.slider("Propagation layers  L", 0, 6, 2)
    feature_type = st.radio("Initial node features", ["Degree", "Random (d=16)"])
    reduction = st.radio("2-D projection", ["PCA", "t-SNE"])


# ── Graph loading ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_graph_data(choice: str, csv_bytes: bytes | None):
    if choice == "Upload CSV":
        df = pd.read_csv(io.BytesIO(csv_bytes))
        G = nx.from_pandas_edgelist(df, df.columns[0], df.columns[1])
    else:
        G = BUILTIN[choice]()
    G = nx.Graph(G)  # ensure undirected, remove multi-edges

    nodes = list(G.nodes())
    pos = nx.spring_layout(G, seed=42)
    A = nx.to_numpy_array(G, nodelist=nodes)

    # Communities / ground-truth labels
    if choice == "Karate Club":
        clubs = [G.nodes[nd].get("club", "?") for nd in nodes]
        unique = sorted(set(clubs))
        colors = [unique.index(c) for c in clubs]
        color_title = "Club"
    else:
        comms = list(nx.community.greedy_modularity_communities(G))
        comm_map = {nd: i for i, c in enumerate(comms) for nd in c}
        colors = [comm_map.get(nd, 0) for nd in nodes]
        color_title = "Community"

    centrality = nx.betweenness_centrality(G)

    return G, nodes, pos, A, colors, color_title, centrality


csv_bytes = uploaded_csv.read() if uploaded_csv else None

if graph_choice == "Upload CSV" and csv_bytes is None:
    st.info("Upload a CSV with two columns: **source**, **target**.")
    st.stop()

G, nodes, pos, A, node_colors, color_title, centrality = load_graph_data(
    graph_choice, csv_bytes
)
n = len(nodes)


# ── GCN propagation ───────────────────────────────────────────────────────────
def make_features(feat_type: str, size: int) -> np.ndarray:
    if feat_type == "Degree":
        degs = np.array([G.degree(nd) for nd in nodes], dtype=float)
        return (degs / degs.max()).reshape(-1, 1)
    rng = np.random.default_rng(42)
    return rng.standard_normal((size, 16))


def gcn_propagate(A: np.ndarray, H: np.ndarray, layers: int) -> np.ndarray:
    """Â = D̃^(-1/2) (A+I) D̃^(-1/2)  ;  H' = Â H  (no weights, pure smoothing)"""
    A_hat = A + np.eye(A.shape[0])
    D_inv_sqrt = np.diag(1.0 / np.sqrt(A_hat.sum(axis=1)))
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
    for _ in range(layers):
        H = A_norm @ H
    return H, A_norm


def project_2d(H: np.ndarray, method: str) -> tuple[np.ndarray, str]:
    H_s = StandardScaler().fit_transform(H)
    n_comp = min(2, H_s.shape[0], H_s.shape[1])

    if method == "PCA" or n < 15:
        pca = PCA(n_components=n_comp)
        emb = pca.fit_transform(H_s)
        if emb.shape[1] < 2:
            emb = np.hstack([emb, np.zeros((emb.shape[0], 2 - emb.shape[1]))])
        label = f"PCA — explained variance {pca.explained_variance_ratio_.sum():.0%}"
    else:
        perp = min(30, max(5, n // 5))
        emb = TSNE(n_components=2, random_state=42, perplexity=perp).fit_transform(H_s)
        label = "t-SNE"
    return emb, label


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Graph Overview", "🧠 GCN Embeddings", "🔍 Node Inspector"])


# ── Tab 1 — Graph overview ────────────────────────────────────────────────────
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nodes", n)
    c2.metric("Edges", G.number_of_edges())
    c3.metric("Avg degree", f"{2 * G.number_of_edges() / n:.2f}")
    c4.metric("Avg clustering", f"{nx.average_clustering(G):.3f}")

    # Edge traces
    ex, ey = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ex += [x0, x1, None]
        ey += [y0, y1, None]

    degrees = [G.degree(nd) for nd in nodes]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ex, y=ey, mode="lines",
        line=dict(width=0.7, color="#bbb"), hoverinfo="none",
    ))
    fig.add_trace(go.Scatter(
        x=[pos[nd][0] for nd in nodes],
        y=[pos[nd][1] for nd in nodes],
        mode="markers+text",
        marker=dict(
            size=[6 + d * 1.5 for d in degrees],
            color=node_colors, colorscale="Viridis", showscale=True,
            colorbar=dict(title=color_title, thickness=12),
            line=dict(width=1, color="white"),
        ),
        text=[str(nd) for nd in nodes],
        textposition="top center", textfont=dict(size=8),
        hovertemplate="<b>Node %{text}</b><br>Degree: %{customdata}<extra></extra>",
        customdata=degrees,
    ))
    fig.update_layout(
        showlegend=False, height=460,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Degree distribution
    deg_counts = pd.Series([d for _, d in G.degree()]).value_counts().sort_index()
    fig_d = go.Figure(go.Bar(
        x=deg_counts.index, y=deg_counts.values, marker_color="#3b82f6",
    ))
    fig_d.update_layout(
        title="Degree distribution", xaxis_title="Degree", yaxis_title="Count",
        height=220, margin=dict(l=0, r=0, t=36, b=0),
    )
    st.plotly_chart(fig_d, use_container_width=True)


# ── Tab 2 — GCN Embeddings ────────────────────────────────────────────────────
with tab2:
    st.markdown(
        r"Each node's embedding is computed by applying $\hat{A}^L$ to the feature matrix, "
        r"where $\hat{A} = \tilde{D}^{-1/2}(A+I)\tilde{D}^{-1/2}$. "
        f"With **L = {n_layers}**, every node aggregates its **{n_layers}-hop neighbourhood**."
    )

    H0 = make_features(feature_type, n)
    H, _ = gcn_propagate(A, H0, n_layers)
    emb, proj_label = project_2d(H, reduction)

    fig_e = go.Figure(go.Scatter(
        x=emb[:, 0], y=emb[:, 1], mode="markers+text",
        marker=dict(
            size=9, color=node_colors, colorscale="Viridis", showscale=True,
            colorbar=dict(title=color_title, thickness=12),
            line=dict(width=1, color="white"),
        ),
        text=[str(nd) for nd in nodes],
        textposition="top center", textfont=dict(size=8),
        hovertemplate="<b>Node %{text}</b><extra></extra>",
    ))
    fig_e.update_layout(
        title=proj_label, height=460,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_e, use_container_width=True)

    # Smoothing curve: track how much node features change layer by layer
    st.markdown("**Feature smoothing** — mean pairwise cosine similarity across layers "
                "(higher = more homogeneous representations):")

    rng_ref = np.random.default_rng(42)
    H_track = make_features(feature_type, n)
    A_hat = A + np.eye(n)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(A_hat.sum(axis=1)))
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt

    similarities = []
    H_l = H_track.copy()
    for layer in range(n_layers + 1):
        norms = np.linalg.norm(H_l, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        H_norm = H_l / norms
        sim = float((H_norm @ H_norm.T).mean())
        similarities.append(sim)
        if layer < n_layers:
            H_l = A_norm @ H_l

    fig_sim = go.Figure(go.Scatter(
        x=list(range(n_layers + 1)), y=similarities,
        mode="lines+markers", line=dict(color="#3b82f6", width=2),
        marker=dict(size=7),
    ))
    fig_sim.update_layout(
        xaxis_title="Layer", yaxis_title="Mean cosine similarity",
        height=200, margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_sim, use_container_width=True)


# ── Tab 3 — Node Inspector ────────────────────────────────────────────────────
with tab3:
    selected = st.selectbox("Select a node", nodes)
    neighbors = sorted(G.neighbors(selected), key=str)

    c1, c2, c3 = st.columns(3)
    c1.metric("Degree", G.degree(selected))
    c2.metric("Clustering coeff.", f"{nx.clustering(G, selected):.3f}")
    c3.metric("Betweenness", f"{centrality[selected]:.4f}")

    st.markdown(
        f"**Neighbours ({len(neighbors)}):** "
        + ", ".join(f"`{nb}`" for nb in neighbors)
    )

    # 1-hop subgraph
    sub_nodes = [selected] + neighbors
    sub_G = G.subgraph(sub_nodes)
    sub_pos = nx.spring_layout(sub_G, seed=42)

    se, sey = [], []
    for u, v in sub_G.edges():
        x0, y0 = sub_pos[u]
        x1, y1 = sub_pos[v]
        se += [x0, x1, None]
        sey += [y0, y1, None]

    s_colors = ["#ef4444" if nd == selected else "#3b82f6" for nd in sub_nodes]

    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(
        x=se, y=sey, mode="lines",
        line=dict(width=1.5, color="#aaa"), hoverinfo="none",
    ))
    fig_s.add_trace(go.Scatter(
        x=[sub_pos[nd][0] for nd in sub_nodes],
        y=[sub_pos[nd][1] for nd in sub_nodes],
        mode="markers+text",
        marker=dict(size=26, color=s_colors, line=dict(width=2, color="white")),
        text=[str(nd) for nd in sub_nodes],
        textposition="middle center", textfont=dict(size=9, color="white"),
        hoverinfo="text",
    ))
    fig_s.update_layout(
        showlegend=False, height=380,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    st.plotly_chart(fig_s, use_container_width=True)
