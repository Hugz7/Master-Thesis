# -*- coding: utf-8 -*-
"""
Visualisations pour le module Backtest Lead-Lag
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict
from backtest import Trade, BacktestResult


def create_equity_curve(result: BacktestResult, initial_capital: float, template: str = "plotly_white") -> go.Figure:
    """Courbe d'equity avec benchmark buy & hold."""
    equity = result.equity_curve
    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3],
                        shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=("Equity Curve", "Drawdown (%)"))

    # Equity curve
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        mode='lines', name='Strategie Lead-Lag',
        line=dict(color='#2196F3', width=2),
        fill='tonexty',
        fillcolor='rgba(33, 150, 243, 0.05)',
        hovertemplate='%{x}<br>Equity: $%{y:,.0f}<extra></extra>'
    ), row=1, col=1)

    # Ligne capital initial
    fig.add_hline(y=initial_capital, line_dash="dash", line_color="gray",
                  annotation_text=f"Capital initial: ${initial_capital:,.0f}", row=1, col=1)

    # Drawdown
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max * 100
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        mode='lines', name='Drawdown',
        line=dict(color='#f44336', width=1),
        fill='tozeroy', fillcolor='rgba(244, 67, 54, 0.15)',
        hovertemplate='%{x}<br>DD: %{y:.2f}%<extra></extra>'
    ), row=2, col=1)

    fig.update_layout(
        template=template, height=600,
        title={'text': "📈 Equity Curve & Drawdown", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18}},
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(tickprefix="$", tickformat=",.0f"),
        yaxis2=dict(ticksuffix="%")
    )
    return fig


def create_trades_scatter(result: BacktestResult, template: str = "plotly_white") -> go.Figure:
    """Scatter plot de tous les trades : PnL vs temps."""
    completed = [t for t in result.trades if t.pnl is not None]
    if not completed:
        return go.Figure()

    winners = [t for t in completed if t.pnl > 0]
    losers = [t for t in completed if t.pnl <= 0]

    fig = go.Figure()

    if winners:
        fig.add_trace(go.Scatter(
            x=[t.exit_time for t in winners],
            y=[t.pnl for t in winners],
            mode='markers', name='Gagnants',
            marker=dict(color='#4CAF50', size=10, symbol='circle',
                        line=dict(color='white', width=1)),
            customdata=[[t.pair, t.direction, t.exit_reason, t.pnl_pct] for t in winners],
            hovertemplate='<b>%{customdata[0]}</b><br>Direction: %{customdata[1]}<br>PnL: $%{y:.2f} (%{customdata[3]:.2f}%)<br>Raison: %{customdata[2]}<extra></extra>'
        ))

    if losers:
        fig.add_trace(go.Scatter(
            x=[t.exit_time for t in losers],
            y=[t.pnl for t in losers],
            mode='markers', name='Perdants',
            marker=dict(color='#f44336', size=10, symbol='circle',
                        line=dict(color='white', width=1)),
            customdata=[[t.pair, t.direction, t.exit_reason, t.pnl_pct] for t in losers],
            hovertemplate='<b>%{customdata[0]}</b><br>Direction: %{customdata[1]}<br>PnL: $%{y:.2f} (%{customdata[3]:.2f}%)<br>Raison: %{customdata[2]}<extra></extra>'
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        template=template, height=450,
        title={'text': "🎯 PnL par Trade", 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="Date", yaxis_title="PnL (USD)",
        yaxis=dict(tickprefix="$"),
        hovermode='closest'
    )
    return fig


def create_pnl_distribution(result: BacktestResult, template: str = "plotly_white") -> go.Figure:
    """Distribution des PnL."""
    completed = [t for t in result.trades if t.pnl is not None]
    if not completed:
        return go.Figure()

    pnls = [t.pnl for t in completed]
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=pnls, nbinsx=30,
        marker_color=['rgba(76, 175, 80, 0.7)' if p > 0 else 'rgba(244, 67, 54, 0.7)' for p in pnls],
        name='Distribution PnL',
        hovertemplate='PnL: $%{x:.0f}<br>Freq: %{y}<extra></extra>'
    ))

    fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=2)
    fig.add_vline(x=np.mean(pnls), line_dash="dot", line_color="blue",
                  annotation_text=f"Moy: ${np.mean(pnls):.0f}", annotation_position="top right")

    fig.update_layout(
        template=template, height=400,
        title={'text': "📊 Distribution des PnL", 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="PnL (USD)", yaxis_title="Frequence",
        showlegend=False
    )
    return fig


def create_performance_by_pair(result: BacktestResult, template: str = "plotly_white") -> go.Figure:
    """PnL cumule par paire de trading."""
    completed = [t for t in result.trades if t.pnl is not None]
    if not completed:
        return go.Figure()

    pair_pnl = {}
    for t in completed:
        pair_pnl[t.pair] = pair_pnl.get(t.pair, 0) + t.pnl

    pairs = sorted(pair_pnl.items(), key=lambda x: x[1])
    names = [p[0] for p in pairs]
    values = [p[1] for p in pairs]
    colors = ['rgba(76, 175, 80, 0.7)' if v > 0 else 'rgba(244, 67, 54, 0.7)' for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation='h',
        marker_color=colors,
        text=[f'${v:.0f}' for v in values],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>PnL: $%{x:.2f}<extra></extra>'
    ))

    fig.add_vline(x=0, line_dash="dash", line_color="black")

    fig.update_layout(
        template=template, height=max(400, len(pairs) * 40),
        title={'text': "📊 PnL par Paire", 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="PnL cumule (USD)", yaxis_title="",
        xaxis=dict(tickprefix="$"),
        showlegend=False, margin=dict(l=180, r=100)
    )
    return fig


def create_monthly_returns_heatmap(result: BacktestResult, template: str = "plotly_white") -> go.Figure:
    """Heatmap des rendements mensuels."""
    equity = result.equity_curve
    if equity.empty:
        return go.Figure()

    monthly = equity.resample('ME').last().pct_change().dropna() * 100
    if monthly.empty:
        return go.Figure()

    monthly_df = monthly.to_frame('return')
    monthly_df['month'] = monthly_df.index.month
    monthly_df['year'] = monthly_df.index.year

    pivot = monthly_df.pivot_table(values='return', index='year', columns='month', aggfunc='sum')
    month_names = ['Jan', 'Fev', 'Mar', 'Avr', 'Mai', 'Jun',
                   'Jul', 'Aou', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot.columns = [month_names[m - 1] for m in pivot.columns]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index.astype(str),
        colorscale='RdYlGn', zmid=0,
        text=np.round(pivot.values, 1),
        texttemplate='%{text}%',
        textfont={"size": 11},
        hovertemplate='%{y} %{x}<br>Rendement: %{z:.2f}%<extra></extra>',
        colorbar=dict(title="Rdt %")
    ))

    fig.update_layout(
        template=template, height=300,
        title={'text': "📅 Rendements Mensuels (%)", 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="", yaxis_title=""
    )
    return fig


def create_exit_reasons_pie(result: BacktestResult, template: str = "plotly_white") -> go.Figure:
    """Pie chart des raisons de sortie."""
    metrics = result.metrics
    exit_reasons = metrics.get('exit_reasons', {})
    if not exit_reasons:
        return go.Figure()

    labels = list(exit_reasons.keys())
    values = list(exit_reasons.values())
    colors = {
        'TP': '#4CAF50', 'SL': '#f44336',
        'TIMEOUT': '#ff9800', 'EOD': '#9e9e9e'
    }
    marker_colors = [colors.get(l, '#2196F3') for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=marker_colors),
        textinfo='label+percent+value',
        hovertemplate='<b>%{label}</b><br>%{value} trades (%{percent})<extra></extra>'
    ))

    fig.update_layout(
        template=template, height=350,
        title={'text': "🎯 Raisons de Sortie", 'x': 0.5, 'xanchor': 'center'},
        showlegend=True
    )
    return fig
