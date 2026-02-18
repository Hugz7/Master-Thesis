# -*- coding: utf-8 -*-
"""
Module de visualisations pour l'application Lead-Lag Analysis
Contient toutes les fonctions de creation de graphiques Plotly
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict


def create_bar_chart(scores, title, template="plotly_white", emoji="📊"):
    colors = ['rgba(76, 175, 80, 0.7)' if x > 0 else 'rgba(244, 67, 54, 0.7)' for x in scores.values]
    fig = go.Figure(go.Bar(
        x=scores.values, y=scores.index, orientation='h',
        marker_color=colors,
        text=[f'{v:.3f}' for v in scores.values],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Score: %{x:.4f}<extra></extra>'
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=2)
    fig.update_layout(
        template=template,
        height=max(400, len(scores) * 40),
        title={'text': f"{emoji} {title}", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'weight': 'bold'}},
        xaxis_title="Score Lead-Lag", yaxis_title="",
        showlegend=False, margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig


def create_top_leaders_followers_chart(mean_scores, template="plotly_white", top_n=5):
    top_leaders = mean_scores.nlargest(top_n).sort_values()
    top_followers = mean_scores.nsmallest(top_n).sort_values()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Top Leaders', y=top_leaders.index, x=top_leaders.values, orientation='h',
        marker=dict(color='rgba(76, 175, 80, 0.8)', line=dict(color='rgba(76, 175, 80, 1)', width=2)),
        text=[f'{v:.3f}' for v in top_leaders.values], textposition='outside',
        hovertemplate='<b>%{y}</b><br>Score: %{x:.4f}<br>Leader<extra></extra>'
    ))
    fig.add_trace(go.Bar(
        name='Top Followers', y=top_followers.index, x=top_followers.values, orientation='h',
        marker=dict(color='rgba(244, 67, 54, 0.8)', line=dict(color='rgba(244, 67, 54, 1)', width=2)),
        text=[f'{v:.3f}' for v in top_followers.values], textposition='outside',
        hovertemplate='<b>%{y}</b><br>Score: %{x:.4f}<br>Follower<extra></extra>'
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=2)
    fig.update_layout(
        template=template,
        title={'text': f"🏆 Top {top_n} Leaders vs Top {top_n} Followers", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},
        xaxis_title="Score Lead-Lag", yaxis_title="",
        barmode='overlay', height=max(500, (top_n * 2) * 50),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=150, r=150)
    )
    return fig


def create_network_graph(scores, threshold=0.3, template="plotly_white"):
    mean_scores = scores.mean()
    G = nx.DiGraph()
    for asset in mean_scores.index:
        G.add_node(asset, score=mean_scores[asset])
    corr_matrix = scores.T.corr()
    for i, asset1 in enumerate(mean_scores.index):
        for j, asset2 in enumerate(mean_scores.index):
            if i != j and abs(corr_matrix.iloc[i, j]) > threshold:
                if mean_scores[asset1] > mean_scores[asset2]:
                    G.add_edge(asset1, asset2, weight=corr_matrix.iloc[i, j])
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode='lines', line=dict(width=1, color='#888'),
            hoverinfo='none', showlegend=False
        ))
    node_x, node_y, node_text, node_color = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x); node_y.append(y)
        score = G.nodes[node]['score']
        node_text.append(f"{node}<br>Score: {score:.3f}")
        node_color.append(score)
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        text=[node for node in G.nodes()], textposition="top center",
        hovertext=node_text, hoverinfo='text',
        marker=dict(size=30, color=node_color, colorscale='RdYlGn',
                    line_width=2, colorbar=dict(title="Score Lead-Lag"),
                    cmin=-1, cmax=1, cmid=0),
        showlegend=False
    )
    fig = go.Figure(data=edge_trace + [node_trace])
    fig.update_layout(
        template=template,
        title={'text': "🔗 Reseau Dirige Lead-Lag<br><sub>Vert = Leader, Rouge = Follower</sub>", 'x': 0.5, 'xanchor': 'center'},
        showlegend=False, hovermode='closest', height=700,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig


def create_score_volatility_scatter(scores, template="plotly_white"):
    mean_scores = scores.mean()
    volatility = scores.std() * np.sqrt(252) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mean_scores, y=volatility, mode='markers+text',
        text=mean_scores.index, textposition="top center",
        marker=dict(size=12, color=mean_scores, colorscale='RdYlGn',
                    showscale=True, colorbar=dict(title="Score<br>Lead-Lag"),
                    line=dict(width=1, color='white'), cmin=-1, cmax=1, cmid=0),
        hovertemplate='<b>%{text}</b><br>Score: %{x:.3f}<br>Volatilite: %{y:.3f}%<extra></extra>'
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Neutre")
    fig.add_hline(y=volatility.median(), line_dash="dash", line_color="lightgray",
                  annotation_text="Volatilite mediane", annotation_position="right")
    fig.update_layout(
        template=template,
        title={'text': "📊 Score Lead-Lag vs Volatilite Annualisee", 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="Score Lead-Lag Moyen", yaxis_title="Volatilite Annualisee (%)",
        height=600, hovermode='closest'
    )
    return fig


def create_score_distribution(scores, template="plotly_white"):
    all_scores = scores.values.flatten()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=all_scores, nbinsx=50,
        marker_color='rgba(33, 150, 243, 0.7)', name='Distribution',
        hovertemplate='Score: %{x:.2f}<br>Frequence: %{y}<extra></extra>'
    ))
    mean_val = np.mean(all_scores)
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                  annotation_text=f"Moyenne: {mean_val:.3f}", annotation_position="top")
    fig.add_vline(x=0, line_dash="dot", line_color="gray",
                  annotation_text="Neutre", annotation_position="bottom")
    fig.update_layout(
        template=template,
        title={'text': "📊 Distribution des Scores Lead-Lag", 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="Score Lead-Lag", yaxis_title="Frequence",
        height=500, showlegend=False
    )
    return fig


def create_levy_area_heatmap(levy_matrix, template="plotly_white"):
    fig = go.Figure(data=go.Heatmap(
        z=levy_matrix.values,
        x=levy_matrix.columns,
        y=levy_matrix.index,
        colorscale='RdBu_r', zmid=0,
        text=np.round(levy_matrix.values, 3),
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='%{y} vs %{x}<br>Aire de Levy: %{z:.4f}<extra></extra>',
        colorbar=dict(title=dict(text="Aire<br>de Levy", side="right"))
    ))
    fig.update_layout(
        template=template,
        title={'text': "📐 Matrice des Aires de Levy<br><sub>A > 0: Premier actif mene, A < 0: Second actif mene</sub>",
               'x': 0.5, 'xanchor': 'center', 'font': {'size': 18}},
        xaxis_title="", yaxis_title="",
        height=max(600, len(levy_matrix) * 30),
        width=max(800, len(levy_matrix) * 30)
    )
    return fig


def create_levy_area_vs_reference_chart(levy_areas, reference_asset, template="plotly_white"):
    sorted_areas = levy_areas.sort_values()
    colors = ['rgba(76, 175, 80, 0.7)' if x > 0 else 'rgba(244, 67, 54, 0.7)' for x in sorted_areas.values]
    fig = go.Figure(go.Bar(
        x=sorted_areas.values, y=sorted_areas.index, orientation='h',
        marker_color=colors,
        text=[f'{v:.3f}' for v in sorted_areas.values],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Aire de Levy: %{x:.4f}<extra></extra>'
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=2)
    fig.update_layout(
        template=template,
        title={'text': f"📐 Aires de Levy par rapport a {reference_asset}<br><sub>A > 0: {reference_asset} mene, A < 0: L'autre actif mene</sub>",
               'x': 0.5, 'xanchor': 'center', 'font': {'size': 18}},
        xaxis_title="Aire de Levy", yaxis_title="",
        height=max(500, len(sorted_areas) * 40),
        showlegend=False, margin=dict(l=150, r=150)
    )
    return fig


def create_levy_area_path_plot(path_data: Dict[str, object], template: str = "plotly_white") -> go.Figure:
    """
    Cree un graphe parametrique 2D de la trajectoire (X(t), Y(t))
    visualisant geometriquement l'Aire de Levy entre deux actifs.
    Inclut la corde (ligne droite debut -> fin) comme reference.
    """
    X = path_data['X']
    Y = path_data['Y']
    dates = path_data['dates']
    levy_area = path_data['levy_area']
    asset_a = path_data['asset_a']
    asset_b = path_data['asset_b']
    n = len(X)

    t_norm = np.linspace(0, 1, n)

    fig = go.Figure()

    # Trace 1 : Zone ombree
    fillcolor = 'rgba(76, 175, 80, 0.10)' if levy_area >= 0 else 'rgba(244, 67, 54, 0.10)'
    poly_x = [0] + list(X) + [0]
    poly_y = [0] + list(Y) + [0]
    fig.add_trace(go.Scatter(
        x=poly_x, y=poly_y,
        fill='toself', fillcolor=fillcolor,
        line=dict(color='rgba(0,0,0,0)'),
        hoverinfo='skip', showlegend=False, name='Aire signee'
    ))

    # Trace 2 : Trajectoire parametrique avec gradient temporel
    fig.add_trace(go.Scatter(
        x=X, y=Y,
        mode='lines+markers',
        line=dict(color='rgba(30, 30, 60, 0.6)', width=1.5),
        marker=dict(
            size=4, color=t_norm, colorscale='Blues',
            showscale=True,
            colorbar=dict(
                title=dict(text="Temps", side="right"),
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=["Debut", "25%", "50%", "75%", "Fin"],
                len=0.6
            ),
        ),
        customdata=np.column_stack([
            np.arange(n),
            [d.strftime('%Y-%m-%d') for d in dates]
        ]),
        hovertemplate=(
            '<b>Jour %{customdata[0]}</b><br>'
            f'{asset_a}: ' + '%{x:.4f}<br>'
            f'{asset_b}: ' + '%{y:.4f}<br>'
            'Date: %{customdata[1]}<extra></extra>'
        ),
        name='Path',
        showlegend=True
    ))

    # Trace 3 : CORDE ORANGE (ligne droite debut -> fin)
    fig.add_trace(go.Scatter(
        x=[X[0], X[-1]],
        y=[Y[0], Y[-1]],
        mode='lines',
        line=dict(color='#FF8C00', width=3),
        name='Chord',
        showlegend=True,
        hovertemplate=(
            f'<b>Corde</b><br>'
            f'Debut: ({X[0]:.4f}, {Y[0]:.4f})<br>'
            f'Fin: ({X[-1]:.4f}, {Y[-1]:.4f})<extra></extra>'
        )
    ))

    # Trace 4 : Marqueur debut (cercle blanc avec contour)
    fig.add_trace(go.Scatter(
        x=[X[0]], y=[Y[0]],
        mode='markers+text',
        marker=dict(size=14, color='white', symbol='circle',
                    line=dict(color='#4CAF50', width=3)),
        text=['Debut'], textposition='top right',
        textfont=dict(size=11, color='#4CAF50'),
        showlegend=False,
        hovertemplate=f'<b>Debut</b><br>{dates[0]:%Y-%m-%d}<extra></extra>'
    ))

    # Trace 5 : Marqueur fin (cercle orange plein)
    fig.add_trace(go.Scatter(
        x=[X[-1]], y=[Y[-1]],
        mode='markers+text',
        marker=dict(size=14, color='#FF8C00', symbol='circle',
                    line=dict(color='white', width=2)),
        text=['Fin'], textposition='bottom left',
        textfont=dict(size=11, color='#FF8C00'),
        showlegend=False,
        hovertemplate=f'<b>Fin</b><br>{dates[-1]:%Y-%m-%d}<extra></extra>'
    ))

    # Fleches directionnelles a 25%, 50%, 75%
    for frac in [0.25, 0.5, 0.75]:
        idx = int(frac * (n - 1))
        if idx + 1 < n:
            fig.add_annotation(
                ax=X[idx], ay=Y[idx],
                x=X[idx + 1], y=Y[idx + 1],
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True, arrowhead=3,
                arrowsize=1.5, arrowwidth=2,
                arrowcolor='rgba(30, 30, 60, 0.7)'
            )

    # Annotation de l'aire de Levy
    if levy_area >= 0:
        direction = f"Anti-horaire ({asset_a} mene)"
        area_color = '#4CAF50'
    else:
        direction = f"Horaire ({asset_b} mene)"
        area_color = '#f44336'

    fig.add_annotation(
        text=(
            f"<b>Aire de Levy = {levy_area:+.4f}</b><br>"
            f"<i>{direction}</i>"
        ),
        xref='paper', yref='paper',
        x=0.02, y=0.98,
        showarrow=False,
        font=dict(size=14, color=area_color),
        bgcolor='rgba(255,255,255,0.85)',
        bordercolor=area_color,
        borderwidth=2,
        borderpad=8,
        align='left'
    )

    fig.update_layout(
        template=template,
        title={
            'text': (
                f"📐 Trajectoire Parametrique : {asset_a} vs {asset_b}"
                f"<br><sub>Aire de Levy = {levy_area:+.4f} | "
                f"{'Anti-horaire = A mene' if levy_area > 0 else 'Horaire = B mene'}</sub>"
            ),
            'x': 0.5, 'xanchor': 'center', 'font': {'size': 18}
        },
        xaxis_title=f"Log-return normalise : {asset_a}",
        yaxis_title=f"Log-return normalise : {asset_b}",
        height=650,
        xaxis=dict(zeroline=True, zerolinecolor='gray', zerolinewidth=1, scaleanchor='y', scaleratio=1),
        yaxis=dict(zeroline=True, zerolinecolor='gray', zerolinewidth=1),
        hovermode='closest',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig