# -*- coding: utf-8 -*-
"""
Application Streamlit Lead-Lag Analysis - FICHIER PRINCIPAL
Analyse des relations lead-lag entre cryptos et marches traditionnels
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
    calculate_levy_areas, calculate_levy_areas_vs_reference, calculate_levy_area_matrix,
    get_levy_path_data
)
from visualizations import (
    create_bar_chart, create_top_leaders_followers_chart,
    create_network_graph, create_score_volatility_scatter,
    create_score_distribution, create_levy_area_heatmap,
    create_levy_area_vs_reference_chart, create_levy_area_path_plot
)
from strategy import generate_trading_strategy
from backtest import LeadLagBacktest, download_hourly_data, INITIAL_CAPITAL, SIGNAL_THRESHOLD, STOP_LOSS_PCT, TAKE_PROFIT_PCT, MAX_HOLDING_HOURS, MIN_LEVY_SCORE
from backtest_visualizations import (
    create_equity_curve, create_trades_scatter, create_pnl_distribution,
    create_performance_by_pair, create_monthly_returns_heatmap, create_exit_reasons_pie
)

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
# STYLES CSS
# ============================================================================

st.markdown("""
    <style>
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
    
    .strategy-box h4,
    .strategy-box p,
    .strategy-box li,
    .strategy-box strong {
        color: #000000 !important;
    }
    
    div[data-testid="stMetric"] * {
        color: var(--text-primary) !important;
    }
    
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
    <h4>🎯 A propos</h4>
    <p>Identifie les relations <strong>lead-lag</strong> entre cryptos et marches traditionnels.</p>
    <ul>
        <li><strong>Leader (&gt; 0)</strong> : Bouge en premier</li>
        <li><strong>Follower (&lt; 0)</strong> : Suit avec retard</li>
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
    selected_cryptos = st.sidebar.multiselect("Selection", list(CRYPTO_PRESETS["🏆 Top 10"]), 
                                               default=['bitcoin', 'ethereum'])

# Actifs traditionnels
st.sidebar.header("2️⃣ Actifs Traditionnels")
trad_choice = st.sidebar.radio("Mode :", ["🎯 Preset", "✏️ Manuel"], key="trad_choice")

if trad_choice == "🎯 Preset":
    trad_preset = st.sidebar.selectbox("Preset", list(TRADITIONAL_PRESETS.keys()))
    selected_traditional = TRADITIONAL_PRESETS[trad_preset]
else:
    selected_traditional = st.sidebar.multiselect("Selection", ['SPY', 'QQQ', 'GLD'], default=['SPY'])

# Parametres
st.sidebar.header("3️⃣ Parametres")
analysis_period = st.sidebar.selectbox("Periode", list(PERIOD_MAP.keys()), index=4)
period_yf = PERIOD_MAP[analysis_period]

window_size = st.sidebar.slider("Fenetre (jours)", 10, 90, 30)
min_assets = st.sidebar.slider("Min. actifs", 2, 10, 3)

with st.sidebar.expander("🔧 Options"):
    dark_mode = st.checkbox("Mode sombre", value=False)
    show_levy = st.checkbox("Aires de Levy", value=True)

run_analysis = st.sidebar.button("🚀 Lancer l'analyse", type="primary", use_container_width=True)

# ============================================================================
# ANALYSE
# ============================================================================

if run_analysis:
    if not selected_cryptos or not selected_traditional:
        st.error("❌ Selectionnez au moins 1 crypto ET 1 actif traditionnel")
        st.stop()
    
    # Telechargement
    st.header("📥 Telechargement")
    with st.spinner("Telechargement..."):
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
        st.success(f"✅ {stats['weekend_filled']} valeurs weekend fillees")
    
    # Nettoyage
    with st.spinner("Nettoyage..."):
        all_prices, _ = clean_and_merge_prices(crypto_aligned, trad_aligned)
    
    if all_prices.empty:
        st.error("❌ Aucune donnee")
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
        
        st.success(f"✅ {len(scores)} fenetres calculees")
        
    except Exception as e:
        st.error(f"❌ Erreur: {e}")
        st.stop()
    
    # Resultats
    mean_scores = scores.mean().sort_values(ascending=False)
    
    st.header("📈 Resultats")
    
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
    
    # Strategie
    st.markdown("---")
    st.header("🎯 Strategie de Trading HFT")
    
    strategy = generate_trading_strategy(
        mean_scores, crypto_prices, traditional_prices, scores, window_size
    )
    
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
    
    st.markdown("### 💡 Strategies")
    
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
            <h4 style='color: #000000; margin-top: 0;'>✨ Opportunites</h4>
        """, unsafe_allow_html=True)
        
        for opp in strategy['opportunities']:
            st.markdown(f"<p style='color: #000000; margin: 10px 0;'>✓ {opp}</p>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="strategy-box" style='background-color: #ffe082; padding: 20px; 
                border-radius: 10px; border: 2px solid #f57c00; margin-top: 20px;'>
        <h4 style='color: #000000; margin-top: 0;'>⚠️ Risques</h4>
    """, unsafe_allow_html=True)
    
    for risk in strategy['risks']:
        st.markdown(f"<p style='color: #000000; margin: 8px 0;'>{risk}</p>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("### 🛡️ Actifs Stables")
    stable_cols = st.columns(len(strategy['stable_assets']))
    for idx, (asset, vol) in enumerate(strategy['stable_assets'].items()):
        with stable_cols[idx]:
            emoji = "🪙" if asset in crypto_prices.columns else "📈"
            st.metric(f"{emoji} {asset}", f"sigma = {vol:.4f}", "Stable")
    
    # ========================================================================
    # VISUALISATIONS
    # ========================================================================
    
    st.markdown("---")
    template = "plotly_dark" if dark_mode else "plotly_white"
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Classements",
        "🕸️ Reseau",
        "📈 Score/Volatilite",
        "📊 Distribution",
        "📐 Aires de Levy",
        "🔄 Trajectoire Parametrique",
        "🧪 Backtest",
        "💾 Export"
    ])
    
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
    
    with tab2:
        st.markdown("### 🕸️ Reseau Lead-Lag")
        threshold = st.slider("Seuil correlation", 0.0, 0.9, 0.3, 0.05)
        try:
            fig_net = create_network_graph(scores, threshold, template)
            st.plotly_chart(fig_net, use_container_width=True)
        except Exception as e:
            st.warning(f"Reseau non disponible: {e}")
    
    with tab3:
        try:
            fig_scatter = create_score_volatility_scatter(scores, template)
            st.plotly_chart(fig_scatter, use_container_width=True)
        except Exception as e:
            st.warning(f"Scatter non disponible: {e}")
    
    with tab4:
        try:
            fig_dist = create_score_distribution(scores, template)
            st.plotly_chart(fig_dist, use_container_width=True)
        except Exception as e:
            st.warning(f"Distribution non disponible: {e}")

    # ---- TAB 5: Aires de Levy (Matrice + Reference) ----
    with tab5:
        if show_levy:
            try:
                with st.spinner("Calcul de la matrice de Levy..."):
                    levy_matrix = calculate_levy_area_matrix(all_prices)

                st.markdown("### 📐 Matrice de Levy")
                st.info("""
                Heatmap complete des aires de Levy entre toutes les paires d'actifs.
                - **Valeur > 0** : La ligne mene la colonne
                - **Valeur < 0** : La colonne mene la ligne
                """)
                fig_levy = create_levy_area_heatmap(levy_matrix, template)
                st.plotly_chart(fig_levy, use_container_width=True)

                st.markdown("---")
                st.markdown("### 📊 Aires vs Actif de Reference")
                st.info("""
                Compare chaque actif par rapport a un actif de reference.
                - **Aire > 0** : L'actif mene la reference
                - **Aire < 0** : L'actif suit la reference
                """)
                ref_asset = st.selectbox("Actif de reference", all_prices.columns.tolist(), key="ref_asset_tab5")
                levy_vs_ref = calculate_levy_areas_vs_reference(all_prices, ref_asset)
                fig_levy_ref = create_levy_area_vs_reference_chart(levy_vs_ref, ref_asset, template)
                st.plotly_chart(fig_levy_ref, use_container_width=True)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Aire max", f"{levy_vs_ref.max():.3f}", levy_vs_ref.idxmax())
                with col2:
                    st.metric("Aire min", f"{levy_vs_ref.min():.3f}", levy_vs_ref.idxmin())
                with col3:
                    st.metric("Aire moyenne", f"{levy_vs_ref.mean():.3f}")

            except Exception as e:
                st.error(f"Erreur calcul Levy: {e}")
        else:
            st.info("Activez 'Aires de Levy' dans les options")

    # ---- TAB 6: Trajectoire Parametrique ----
    with tab6:
        if show_levy:
            st.markdown("### 🔄 Trajectoire Parametrique (Levy's Stochastic Area)")
            st.info("""
            Visualisation geometrique de l'Aire de Levy entre deux actifs.
            La courbe trace (X(t), Y(t)) dans le plan des log-returns normalises.
            - **Boucle anti-horaire** : Aire positive, le premier actif mene
            - **Boucle horaire** : Aire negative, le second actif mene
            """)
            try:
                asset_list = all_prices.columns.tolist()
                col_sel1, col_sel2 = st.columns(2)
                with col_sel1:
                    path_asset_a = st.selectbox("Actif A (axe X)", asset_list, index=0, key="levy_path_asset_a")
                with col_sel2:
                    default_b_index = 1 if len(asset_list) > 1 else 0
                    path_asset_b = st.selectbox("Actif B (axe Y)", asset_list, index=default_b_index, key="levy_path_asset_b")

                if path_asset_a == path_asset_b:
                    st.warning("Veuillez selectionner deux actifs differents.")
                else:
                    with st.spinner("Generation de la trajectoire..."):
                        path_data = get_levy_path_data(all_prices, path_asset_a, path_asset_b)

                    area_val = path_data['levy_area']
                    leader = path_asset_a if area_val > 0 else path_asset_b
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric("Aire de Levy", f"{area_val:+.4f}", delta=f"Leader: {leader}", delta_color="normal" if area_val > 0 else "inverse")
                    with col_m2:
                        st.metric("Jours analyses", f"{len(path_data['X'])}")
                    with col_m3:
                        st.metric("Direction", "Anti-horaire" if area_val > 0 else "Horaire", delta=f"{leader} mene")

                    fig_path = create_levy_area_path_plot(path_data, template)
                    st.plotly_chart(fig_path, use_container_width=True)
            except Exception as e:
                st.error(f"Erreur trajectoire: {e}")
        else:
            st.info("Activez 'Aires de Levy' dans les options")

    # ---- TAB 7: Backtest ----
    with tab7:
        st.markdown("### 🧪 Backtest Lead-Lag - Donnees Horaires")
        st.info("""
        **Strategie** : Quand un leader bouge de +/-1.5% sur une bougie horaire, on entre sur le follower.
        **Exchange** : Binance | **Frais** : 0.1% | **Capital** : $100,000
        """)

        # Parametres backtest dans des colonnes
        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        with col_b1:
            bt_signal = st.slider("Signal (%)", 0.5, 5.0, 1.5, 0.1, key="bt_signal",
                                  help="Mouvement minimum du leader pour declencher un signal")
        with col_b2:
            bt_sl = st.slider("Stop-Loss (%)", 0.5, 5.0, 2.0, 0.1, key="bt_sl")
        with col_b3:
            bt_tp = st.slider("Take-Profit (%)", 1.0, 10.0, 3.0, 0.1, key="bt_tp")
        with col_b4:
            bt_hours = st.slider("Timeout (h)", 1, 12, 3, 1, key="bt_hours",
                                 help="Sortie forcee apres N heures si ni SL ni TP atteint")

        bt_period = st.selectbox("Periode historique", ["30d", "60d", "90d"], index=1, key="bt_period")

        run_backtest = st.button("🚀 Lancer le Backtest", type="primary", use_container_width=True, key="run_backtest")

        if run_backtest:
            with st.spinner("Telechargement des donnees horaires..."):
                try:
                    bt_crypto, bt_trad = download_hourly_data(
                        selected_cryptos, selected_traditional, period=bt_period
                    )
                    st.success(f"Donnees horaires : {len(bt_crypto)} bougies sur {len(bt_crypto.columns) + len(bt_trad.columns)} actifs")
                except Exception as e:
                    st.error(f"Erreur telechargement horaire: {e}")
                    st.stop()

            with st.spinner("Execution du backtest..."):
                try:
                    engine = LeadLagBacktest(
                        crypto_prices=bt_crypto,
                        traditional_prices=bt_trad,
                        levy_scores=mean_scores,
                        initial_capital=INITIAL_CAPITAL,
                        signal_threshold=bt_signal / 100,
                        stop_loss_pct=bt_sl / 100,
                        take_profit_pct=bt_tp / 100,
                        max_holding_hours=bt_hours,
                        min_levy_score=MIN_LEVY_SCORE
                    )
                    bt_result = engine.run(show_progress=False)
                except Exception as e:
                    st.error(f"Erreur backtest: {e}")
                    st.stop()

            m = bt_result.metrics
            if not m:
                st.warning("Aucun trade genere. Essayez de baisser le seuil de signal ou d'augmenter la periode.")
            else:
                # Metriques principales
                st.markdown("#### 📊 Performance")
                mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                ret_color = "normal" if m['total_return_pct'] > 0 else "inverse"
                with mc1:
                    st.metric("Rendement Total", f"{m['total_return_pct']:+.2f}%",
                              delta=f"${m['final_equity'] - INITIAL_CAPITAL:+,.0f}")
                with mc2:
                    st.metric("Sharpe Ratio", f"{m['sharpe_ratio']:.3f}")
                with mc3:
                    st.metric("Sortino Ratio", f"{m['sortino_ratio']:.3f}")
                with mc4:
                    st.metric("Max Drawdown", f"{m['max_drawdown_pct']:.2f}%")
                with mc5:
                    st.metric("Win Rate", f"{m['win_rate_pct']:.1f}%",
                              delta=f"{m['total_trades']} trades")

                mc6, mc7, mc8, mc9 = st.columns(4)
                with mc6:
                    st.metric("Profit Factor", f"{m['profit_factor']:.3f}")
                with mc7:
                    st.metric("Gain moyen", f"${m['avg_win_usd']:+.2f}")
                with mc8:
                    st.metric("Perte moyenne", f"${m['avg_loss_usd']:+.2f}")
                with mc9:
                    st.metric("Frais payes", f"${m['total_fees_usd']:,.2f}")

                # Graphiques
                st.plotly_chart(create_equity_curve(bt_result, INITIAL_CAPITAL, template), use_container_width=True)

                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    st.plotly_chart(create_trades_scatter(bt_result, template), use_container_width=True)
                with col_g2:
                    st.plotly_chart(create_pnl_distribution(bt_result, template), use_container_width=True)

                col_g3, col_g4 = st.columns(2)
                with col_g3:
                    st.plotly_chart(create_performance_by_pair(bt_result, template), use_container_width=True)
                with col_g4:
                    st.plotly_chart(create_exit_reasons_pie(bt_result, template), use_container_width=True)

                st.plotly_chart(create_monthly_returns_heatmap(bt_result, template), use_container_width=True)

                # Tableau des trades
                with st.expander("📋 Detail de tous les trades"):
                    trades_data = []
                    for t in bt_result.trades:
                        if t.pnl is not None:
                            trades_data.append({
                                'Paire': t.pair,
                                'Direction': t.direction,
                                'Entree': t.entry_time,
                                'Sortie': t.exit_time,
                                'Prix entree': f"${t.entry_price:.4f}",
                                'Prix sortie': f"${t.exit_price:.4f}",
                                'PnL ($)': round(t.pnl, 2),
                                'PnL (%)': round(t.pnl_pct, 2),
                                'Raison': t.exit_reason,
                                'Frais ($)': round(t.fees_paid, 2)
                            })
                    if trades_data:
                        df_trades = pd.DataFrame(trades_data)
                        st.dataframe(
                            df_trades.style.applymap(
                                lambda v: 'color: green' if isinstance(v, (int, float)) and v > 0
                                else ('color: red' if isinstance(v, (int, float)) and v < 0 else ''),
                                subset=['PnL ($)', 'PnL (%)']
                            ),
                            use_container_width=True
                        )
                        csv_bt = df_trades.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Telecharger trades CSV", csv_bt,
                                          f"backtest_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv")
        else:
            st.info("Configure les parametres ci-dessus et clique sur **Lancer le Backtest**")

    # ---- TAB 8: Export ----
    with tab8:
        st.markdown("### 💾 Export")
        
        df_export = mean_scores.to_frame("Score")
        df_export["Type"] = df_export.index.map(
            lambda x: "Crypto" if x in crypto_prices.columns else "Tradi"
        )
        df_export["Rang"] = range(1, len(df_export) + 1)
        
        st.dataframe(
            df_export.style.background_gradient(subset=['Score'], cmap='RdYlGn', vmin=-1, vmax=1),
            use_container_width=True
        )
        
        csv = df_export.to_csv().encode('utf-8')
        st.download_button("📥 Telecharger CSV", csv, 
                          f"leadlag_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv")

else:
    st.info("👈 Configurez et cliquez sur **🚀 Lancer l'analyse**")
