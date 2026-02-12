"""
Application Streamlit Lead-Lag Analysis - FICHIER PRINCIPAL
Analyse des relations lead-lag entre cryptos et marchés traditionnels
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import traceback
from datetime import datetime
import sys
import os

# Imports des modules locaux
from utils import (
    CRYPTO_PRESETS, TRADITIONAL_PRESETS, PERIOD_MAP,
    validate_data_quality, download_all_assets_yfinance,
    align_crypto_traditional_data, clean_and_merge_prices,
    calculate_levy_areas, calculate_levy_areas_vs_reference, calculate_levy_area_matrix
)
from visualizations import (
    create_bar_chart, create_top_leaders_followers_chart,
    create_network_graph, create_score_volatility_scatter,
    create_score_distribution, create_levy_area_heatmap,
    create_levy_area_vs_reference_chart
)
from strategy import generate_trading_strategy

# Ajouter chemin pour portfolio_py313
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# ============================================================================
# CONFIGURATION PAGE
# ============================================================================

st.set_page_config(
    page_title="Crypto Lead-Lag Analysis Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# STYLES CSS - TEXTE NOIR SUR FONDS COLORÉS
# ============================================================================

st.markdown("""
    <style>
    /* Support thème clair et sombre */
    :root {
        --text-primary: #1a1a1a;
        --bg-primary: #ffffff;
        --bg-secondary: #f8f9fa;
        --bg-accent: #e8f4f8;
    }
    
    @media (prefers-color-scheme: dark) {
        :root {
            --text-primary: #e0e0e0;
            --bg-primary: #1e1e1e;
            --bg-secondary: #2d2d2d;
            --bg-accent: #1a3a4a;
        }
    }
    
    /* Force BLACK text on colored backgrounds */
    .strategy-box h4,
    .strategy-box p,
    .strategy-box li,
    .strategy-box strong {
        color: #000000 !important;
    }
    
    /* Métriques */
    div[data-testid="stMetric"] * {
        color: var(--text-primary) !important;
    }
    
    /* Texte général */
    .main, .stMarkdown, p, li, h1, h2, h3 {
        color: var(--text-primary) !important;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================

st.title("📊 Analyse Lead-Lag : Crypto vs Actifs Traditionnels")

st.markdown("""
<div class="custom-box">
    <h4>🎯 À propos</h4>
    <p>Identifie les relations <strong>lead-lag</strong> entre cryptos et marchés traditionnels.</p>
    <ul>
        <li><strong>Leader (> 0)</strong> : Bouge en premier</li>
        <li><strong>Follower (< 0)</strong> : Suit avec retard</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("⚙️ Configuration")

# Cryptos
st.sidebar.header("1️⃣ Cryptomonnaies")
crypto_choice = st.sidebar.radio("Mode :", ["🎯 Preset", "✏️ Manuel"], key="crypto_choice")

if crypto_choice == "🎯 Preset":
    crypto_preset = st.sidebar.selectbox("Preset", list(CRYPTO_PRESETS.keys()))
    selected_cryptos = CRYPTO_PRESETS[crypto_preset]
else:
    selected_cryptos = st.sidebar.multiselect("Sélection", list(CRYPTO_PRESETS["🏆 Top 10"]), 
                                               default=['bitcoin', 'ethereum'])

# Actifs traditionnels
st.sidebar.header("2️⃣ Actifs Traditionnels")
trad_choice = st.sidebar.radio("Mode :", ["🎯 Preset", "✏️ Manuel"], key="trad_choice")

if trad_choice == "🎯 Preset":
    trad_preset = st.sidebar.selectbox("Preset", list(TRADITIONAL_PRESETS.keys()))
    selected_traditional = TRADITIONAL_PRESETS[trad_preset]
else:
    selected_traditional = st.sidebar.multiselect("Sélection", ['SPY', 'QQQ', 'GLD'], default=['SPY'])

# Paramètres
st.sidebar.header("3️⃣ Paramètres")
analysis_period = st.sidebar.selectbox("Période", list(PERIOD_MAP.keys()), index=4)
period_yf = PERIOD_MAP[analysis_period]

window_size = st.sidebar.slider("Fenêtre (jours)", 10, 90, 30)
min_assets = st.sidebar.slider("Min. actifs", 2, 10, 3)

with st.sidebar.expander("🔧 Options"):
    dark_mode = st.checkbox("Mode sombre", value=False)
    show_levy = st.checkbox("Aires de Lévy", value=True)

run_analysis = st.sidebar.button("🚀 Lancer l'analyse", type="primary", use_container_width=True)

# ============================================================================
# ANALYSE
# ============================================================================

if run_analysis:
    if not selected_cryptos or not selected_traditional:
        st.error("❌ Sélectionnez au moins 1 crypto ET 1 actif traditionnel")
        st.stop()
    
    # Téléchargement
    st.header("📥 Téléchargement")
    with st.spinner("Téléchargement..."):
        crypto_prices, traditional_prices, log = download_all_assets_yfinance(
            selected_cryptos, selected_traditional, period_yf
        )
    
    # Alignement
    st.header("🔗 Alignement")
    with st.spinner("Alignement weekend..."):
        crypto_aligned, trad_aligned, stats = align_crypto_traditional_data(
            crypto_prices, traditional_prices
        )
    
    if stats['weekend_filled'] > 0:
        st.success(f"✅ {stats['weekend_filled']} valeurs weekend fillées")
    
    # Nettoyage
    with st.spinner("Nettoyage..."):
        all_prices, _ = clean_and_merge_prices(crypto_aligned, trad_aligned)
    
    if all_prices.empty:
        st.error("❌ Aucune donnée")
        st.stop()
    
    st.success(f"✅ {len(all_prices)} jours × {len(all_prices.columns)} actifs")
    
    # Analyse Lead-Lag
    st.header("🧮 Analyse Lead-Lag")
    try:
        with st.spinner("Calcul..."):
            from portfolio_py313 import LeadLagPortfolio
            
            portfolio = LeadLagPortfolio(all_prices)
            portfolio.generate_matrices_and_networks(
                window_size=window_size, min_assets=min_assets, 
                show_progress=False, backend='manual'
            )
            scores = portfolio.calculate_global_scores(show_progress=False)
            rankings = portfolio.rank_assets_global(selection_pct=0.3)
        
        st.success(f"✅ {len(scores)} fenêtres calculées")
        
    except Exception as e:
        st.error(f"❌ Erreur: {e}")
        st.stop()
    
    # Résultats
    mean_scores = scores.mean().sort_values(ascending=False)
    
    st.header("📈 Résultats")
    
    cols = st.columns(4)
    with cols[0]:
        st.metric("Score max", f"{mean_scores.max():.3f}", mean_scores.idxmax())
    with cols[1]:
        st.metric("Score min", f"{mean_scores.min():.3f}", mean_scores.idxmin())
    with cols[2]:
        crypto_leaders = sum(1 for a in mean_scores.index if a in crypto_prices.columns and mean_scores[a] > 0)
        st.metric("Cryptos leaders", f"{crypto_leaders}/{len([c for c in crypto_prices.columns if c in mean_scores.index])}")
    with cols[3]:
        trad_leaders = sum(1 for a in mean_scores.index if a in traditional_prices.columns and mean_scores[a] > 0)
        st.metric("Tradis leaders", f"{trad_leaders}/{len([t for t in traditional_prices.columns if t in mean_scores.index])}")
    
    # ========================================================================
    # STRATÉGIE avec texte NOIR
    # ========================================================================
    
    st.markdown("---")
    st.header("🎯 Stratégie de Trading HFT")
    
    strategy = generate_trading_strategy(
        mean_scores, crypto_prices, traditional_prices, scores, window_size
    )
    
    # Signal
    signal_colors = {
        'CRYPTO_LEAD': '#4CAF50',
        'MARKET_LEAD': '#2196F3',
        'MIXED': '#ff9800'
    }
    color = signal_colors.get(strategy['signal'], '#999')
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {color}33, {color}22); 
                padding: 25px; border-radius: 15px; border-left: 5px solid {color};'>
        <h3 style='margin: 0; color: #000000;'>Signal: {strategy['signal']}</h3>
        <p style='font-size: 18px; color: #000000;'>
            <strong>Confiance: {strategy['confidence']}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Leaders/Followers avec TEXTE NOIR
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        st.markdown("### 🏆 Top 3 Leaders")
        for asset, score in list(strategy['leaders'].items()):
            emoji = "🪙" if asset in crypto_prices.columns else "📈"
            st.markdown(f"""
            <div style='background-color: #c8e6c9; padding: 10px; border-radius: 8px; 
                        margin: 5px 0; border-left: 4px solid #4CAF50;'>
                <strong style='color: #000000;'>{emoji} {asset}</strong><br>
                <span style='color: #000000; font-size: 18px; font-weight: bold;'>
                    {score:+.4f}
                </span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🔻 Top 3 Followers")
        for asset, score in list(strategy['followers'].items()):
            emoji = "🪙" if asset in crypto_prices.columns else "📈"
            st.markdown(f"""
            <div style='background-color: #ffcdd2; padding: 10px; border-radius: 8px; 
                        margin: 5px 0; border-left: 4px solid #f44336;'>
                <strong style='color: #000000;'>{emoji} {asset}</strong><br>
                <span style='color: #000000; font-size: 18px; font-weight: bold;'>
                    {score:+.4f}
                </span>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("### 📊 Top 5")
        try:
            fig_top = create_top_leaders_followers_chart(
                mean_scores, "plotly_dark" if dark_mode else "plotly_white", top_n=5
            )
            st.plotly_chart(fig_top, use_container_width=True)
        except Exception as e:
            st.error(f"Erreur: {e}")
    
    # Stratégies avec TEXTE NOIR
    st.markdown("### 💡 Stratégies")
    
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        st.markdown("""
        <div class="strategy-box" style='background-color: #b3d9ff; padding: 20px; 
                    border-radius: 10px; border: 2px solid #1976d2;'>
            <h4 style='color: #000000; margin-top: 0;'>📋 Actions</h4>
        """, unsafe_allow_html=True)
        
        for rec in strategy['recommendations']:
            st.markdown(f"<p style='color: #000000; margin: 10px 0;'>{rec}</p>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col_s2:
        st.markdown("""
        <div class="strategy-box" style='background-color: #a5d6a7; padding: 20px; 
                    border-radius: 10px; border: 2px solid #388e3c;'>
            <h4 style='color: #000000; margin-top: 0;'>✨ Opportunités</h4>
        """, unsafe_allow_html=True)
        
        for opp in strategy['opportunities']:
            st.markdown(f"<p style='color: #000000; margin: 10px 0;'>✓ {opp}</p>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Risques avec TEXTE NOIR
    st.markdown("""
    <div class="strategy-box" style='background-color: #ffe082; padding: 20px; 
                border-radius: 10px; border: 2px solid #f57c00; margin-top: 20px;'>
        <h4 style='color: #000000; margin-top: 0;'>⚠️ Risques</h4>
    """, unsafe_allow_html=True)
    
    for risk in strategy['risks']:
        st.markdown(f"<p style='color: #000000; margin: 8px 0;'>{risk}</p>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Actifs stables
    st.markdown("### 🛡️ Actifs Stables")
    stable_cols = st.columns(len(strategy['stable_assets']))
    for idx, (asset, vol) in enumerate(strategy['stable_assets'].items()):
        with stable_cols[idx]:
            emoji = "🪙" if asset in crypto_prices.columns else "📈"
            st.metric(f"{emoji} {asset}", f"σ = {vol:.4f}", "Stable")
    
    # ========================================================================
    # VISUALISATIONS
    # ========================================================================
    
    st.markdown("---")
    template = "plotly_dark" if dark_mode else "plotly_white"
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Classements",
        "🕸️ Réseau",
        "📈 Score/Volatilité",
        "📊 Distribution",
        "📐 Aires de Lévy",
        "💾 Export"
    ])
    
    # TAB 1: Classements
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            crypto_scores = mean_scores[[c for c in crypto_prices.columns if c in mean_scores.index]]
            fig_crypto = create_bar_chart(crypto_scores, "Cryptomonnaies", template, "🪙")
            st.plotly_chart(fig_crypto, use_container_width=True)
        
        with col2:
            trad_scores = mean_scores[[t for t in traditional_prices.columns if t in mean_scores.index]]
            fig_trad = create_bar_chart(trad_scores, "Actifs Traditionnels", template, "📈")
            st.plotly_chart(fig_trad, use_container_width=True)
    
    # TAB 2: Réseau
    with tab2:
        st.markdown("### 🕸️ Réseau Lead-Lag")
        threshold = st.slider("Seuil corrélation", 0.0, 0.9, 0.3, 0.05)
        try:
            fig_net = create_network_graph(scores, threshold, template)
            st.plotly_chart(fig_net, use_container_width=True)
        except:
            st.warning("Réseau non disponible")
    
    # TAB 3: Score/Volatilité
    with tab3:
        try:
            fig_scatter = create_score_volatility_scatter(scores, template)
            st.plotly_chart(fig_scatter, use_container_width=True)
        except:
            st.warning("Scatter non disponible")
    
    # TAB 4: Distribution
    with tab4:
        try:
            fig_dist = create_score_distribution(scores, template)
            st.plotly_chart(fig_dist, use_container_width=True)
        except:
            st.warning("Distribution non disponible")
    
    # TAB 5: Aires de Lévy
    with tab5:
        if show_levy:
            st.markdown("### 📐 Aires de Lévy")
            st.info("""
            Les Aires de Lévy mesurent la relation géométrique entre deux actifs.
            - **A > 0** : Premier actif mène
            - **A < 0** : Second actif mène
            """)
            
            try:
                with st.spinner("Calcul Aires de Lévy..."):
                    levy_matrix = calculate_levy_area_matrix(all_prices)
                
                # Matrice complète
                fig_levy = create_levy_area_heatmap(levy_matrix, template)
                st.plotly_chart(fig_levy, use_container_width=True)
                
                # Vs référence
                st.markdown("### 📐 Aires vs Actif de Référence")
                ref_asset = st.selectbox("Référence", all_prices.columns.tolist())
                
                levy_vs_ref = calculate_levy_areas_vs_reference(all_prices, ref_asset)
                fig_levy_ref = create_levy_area_vs_reference_chart(levy_vs_ref, ref_asset, template)
                st.plotly_chart(fig_levy_ref, use_container_width=True)
                
                # Stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Aire max", f"{levy_vs_ref.max():.3f}", levy_vs_ref.idxmax())
                with col2:
                    st.metric("Aire min", f"{levy_vs_ref.min():.3f}", levy_vs_ref.idxmin())
                with col3:
                    st.metric("Aire moyenne", f"{levy_vs_ref.mean():.3f}")
                
            except Exception as e:
                st.error(f"Erreur calcul Lévy: {e}")
        else:
            st.info("Activez 'Aires de Lévy' dans les options")
    
    # TAB 6: Export
    with tab6:
        st.markdown("### 💾 Export")
        
        df_export = mean_scores.to_frame("Score")
        df_export["Type"] = df_export.index.map(
            lambda x: "🪙 Crypto" if x in crypto_prices.columns else "📈 Tradi"
        )
        df_export["Rang"] = range(1, len(df_export) + 1)
        
        st.dataframe(
            df_export.style.background_gradient(subset=['Score'], cmap='RdYlGn', vmin=-1, vmax=1),
            use_container_width=True
        )
        
        csv = df_export.to_csv().encode('utf-8')
        st.download_button("📥 Télécharger CSV", csv, 
                          f"leadlag_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv")

else:
    st.info("👈 Configurez et cliquez sur **🚀 Lancer l'analyse**")
