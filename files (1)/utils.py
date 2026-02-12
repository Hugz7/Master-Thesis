"""
Module utilitaire pour l'application Lead-Lag Analysis
Contient les fonctions de validation, téléchargement et calculs mathématiques
"""

import pandas as pd
import numpy as np
from typing import Any, Tuple, Dict, List
import yfinance as yf


# ============================================================================
# CONSTANTES
# ============================================================================

CRYPTO_TICKERS_MAP = {
    'bitcoin': 'BTC-USD', 'ethereum': 'ETH-USD', 'binancecoin': 'BNB-USD',
    'ripple': 'XRP-USD', 'cardano': 'ADA-USD', 'solana': 'SOL-USD',
    'polkadot': 'DOT-USD', 'dogecoin': 'DOGE-USD', 'avalanche-2': 'AVAX-USD',
    'chainlink': 'LINK-USD', 'litecoin': 'LTC-USD', 'uniswap': 'UNI-USD',
    'stellar': 'XLM-USD', 'monero': 'XMR-USD', 'polygon': 'MATIC-USD',
    'cosmos': 'ATOM-USD', 'algorand': 'ALGO-USD', 'tron': 'TRX-USD'
}

CRYPTO_PRESETS = {
    "🏆 Top 10": ['bitcoin', 'ethereum', 'binancecoin', 'ripple', 'cardano', 
                  'solana', 'polkadot', 'dogecoin', 'avalanche-2', 'chainlink'],
    "💎 DeFi Focus": ['ethereum', 'uniswap', 'chainlink', 'avalanche-2'],
    "🔗 Layer 1": ['bitcoin', 'ethereum', 'cardano', 'solana', 'polkadot'],
    "⚡ Performance": ['solana', 'avalanche-2', 'polygon', 'cosmos'],
    "🪙 Majors": ['bitcoin', 'ethereum', 'binancecoin']
}

TRADITIONAL_PRESETS = {
    "📊 S&P 500": ['SPY'],
    "🖥️ Nasdaq 100": ['QQQ'],
    "🏛️ Dow Jones": ['DIA'],
    "💰 Or": ['GLD'],
    "🥈 Argent": ['SLV'],
    "🛢️ Pétrole": ['USO'],
    "📈 Indices majeurs": ['SPY', 'QQQ', 'DIA'],
    "🚀 Tech Giants": ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA'],
    "⚡ Semiconducteurs": ['NVDA', 'AMD', 'INTC', 'TSM'],
    "💎 Matières premières": ['GLD', 'SLV', 'USO', 'DBA'],
    "🌐 Mix complet": ['SPY', 'QQQ', 'GLD', 'AAPL', 'MSFT'],
}

PERIOD_MAP = {
    "1 mois": "1mo",
    "3 mois": "3mo", 
    "6 mois": "6mo",
    "1 an": "1y",
    "2 ans": "2y",
    "5 ans": "5y"
}


# ============================================================================
# VALIDATION DES DONNÉES
# ============================================================================

def validate_data_quality(df: pd.DataFrame, name: str = "Dataset") -> Dict[str, Any]:
    """
    Valide la qualité des données avec statistiques détaillées
    """
    if df.empty:
        return {
            'valid': False,
            'message': f"{name} est vide",
            'stats': {}
        }
    
    stats = {
        'rows': len(df),
        'cols': len(df.columns),
        'missing_pct': (df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100),
        'date_range': (df.index[0].date(), df.index[-1].date()) if len(df) > 0 else None,
        'columns': list(df.columns)
    }
    
    # Validation
    if stats['rows'] < 20:
        return {
            'valid': False,
            'message': f"{name} a trop peu de lignes ({stats['rows']} < 20)",
            'stats': stats
        }
    
    if stats['missing_pct'] > 30:
        return {
            'valid': False,
            'message': f"{name} a trop de données manquantes ({stats['missing_pct']:.1f}%)",
            'stats': stats
        }
    
    return {
        'valid': True,
        'message': f"{name} valide ✓",
        'stats': stats
    }


# ============================================================================
# TÉLÉCHARGEMENT DONNÉES
# ============================================================================

def download_all_assets_yfinance(
    crypto_tickers: List[str], 
    traditional_tickers: List[str], 
    period: str = '2y'
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """
    Télécharge les données avec gestion d'erreurs améliorée
    
    Returns:
        Tuple[crypto_prices, traditional_prices, download_log]
    """
    download_log = {
        'crypto_requested': len(crypto_tickers),
        'trad_requested': len(traditional_tickers),
        'crypto_success': 0,
        'trad_success': 0,
        'errors': []
    }
    
    # Conversion cryptos
    crypto_yahoo = [CRYPTO_TICKERS_MAP.get(c.lower(), c) for c in crypto_tickers]
    all_tickers = list(set(crypto_yahoo + traditional_tickers))
    
    try:
        # Téléchargement groupé
        data = yf.download(
            all_tickers, 
            period=period, 
            progress=False,
            auto_adjust=True,
            threads=True
        )
        
        if data.empty:
            download_log['errors'].append("Aucune donnée retournée par Yahoo Finance")
            return pd.DataFrame(), pd.DataFrame(), download_log
        
        # Extraction prix
        if isinstance(data.columns, pd.MultiIndex):
            prices = data.get('Adj Close', data.get('Close', data))
        else:
            prices = data
        
        # Nettoyage noms colonnes
        prices.columns = [col.replace('-USD', '').upper() for col in prices.columns]
        
        # Séparation crypto / traditionnel
        crypto_symbols = [t.replace('-USD', '').upper() for t in crypto_yahoo]
        crypto_cols = [c for c in crypto_symbols if c in prices.columns]
        trad_cols = [c for c in prices.columns if c not in crypto_symbols]
        
        crypto_prices = prices[crypto_cols] if crypto_cols else pd.DataFrame()
        trad_prices = prices[trad_cols] if trad_cols else pd.DataFrame()
        
        download_log['crypto_success'] = len(crypto_cols)
        download_log['trad_success'] = len(trad_cols)
        
        return crypto_prices, trad_prices, download_log
        
    except Exception as e:
        download_log['errors'].append(f"Erreur téléchargement: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), download_log


# ============================================================================
# ALIGNEMENT ET NETTOYAGE
# ============================================================================

def align_crypto_traditional_data(
    crypto_prices: pd.DataFrame,
    trad_prices: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Aligne les données crypto (7j/7) avec les données traditionnelles (5j/7)
    en forward-fillant les prix traditionnels sur les weekends
    """
    alignment_stats = {
        'crypto_initial': len(crypto_prices) if not crypto_prices.empty else 0,
        'trad_initial': len(trad_prices) if not trad_prices.empty else 0,
        'weekend_filled': 0
    }
    
    if crypto_prices.empty or trad_prices.empty:
        return crypto_prices, trad_prices, alignment_stats
    
    # Créer un index commun basé sur les cryptos (7j/7)
    full_index = crypto_prices.index
    
    # Compter les NaN avant alignement
    initial_nans = trad_prices.reindex(full_index).isna().sum().sum()
    
    # Réindexer les données traditionnelles sur l'index des cryptos
    trad_aligned = trad_prices.reindex(full_index)
    
    # Forward fill pour les weekends (max 2 jours)
    trad_aligned = trad_aligned.ffill(limit=2)
    
    # Compter les valeurs fillées (NaN avant - NaN après)
    final_nans = trad_aligned.isna().sum().sum()
    alignment_stats['weekend_filled'] = initial_nans - final_nans
    
    return crypto_prices, trad_aligned, alignment_stats


def clean_and_merge_prices(
    crypto_prices: pd.DataFrame,
    trad_prices: pd.DataFrame,
    min_valid_pct: float = 0.65
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fusionne et nettoie les prix avec statistiques
    """
    cleaning_stats = {
        'initial_crypto_cols': len(crypto_prices.columns) if not crypto_prices.empty else 0,
        'initial_trad_cols': len(trad_prices.columns) if not trad_prices.empty else 0,
        'initial_rows': 0,
        'final_rows': 0,
        'removed_cols': [],
        'fill_operations': 0
    }
    
    # Fusion
    if crypto_prices.empty and trad_prices.empty:
        return pd.DataFrame(), cleaning_stats
    
    if crypto_prices.empty:
        all_prices = trad_prices.copy()
    elif trad_prices.empty:
        all_prices = crypto_prices.copy()
    else:
        all_prices = pd.concat([crypto_prices, trad_prices], axis=1)
    
    all_prices = all_prices.sort_index()
    cleaning_stats['initial_rows'] = len(all_prices)
    
    # Suppression colonnes avec trop de NaN
    valid_threshold = len(all_prices) * (1 - min_valid_pct)
    cols_to_drop = all_prices.columns[all_prices.isna().sum() > valid_threshold]
    cleaning_stats['removed_cols'] = list(cols_to_drop)
    all_prices = all_prices.drop(columns=cols_to_drop)
    
    # Forward fill puis backward fill (limité)
    all_prices = all_prices.ffill(limit=4)
    cleaning_stats['fill_operations'] += 1
    all_prices = all_prices.bfill(limit=2)
    cleaning_stats['fill_operations'] += 1
    
    # Suppression lignes avec trop de NaN
    min_valid_cols = int(len(all_prices.columns) * min_valid_pct)
    all_prices = all_prices.dropna(thresh=min_valid_cols)
    
    cleaning_stats['final_rows'] = len(all_prices)
    
    return all_prices, cleaning_stats


# ============================================================================
# CALCUL AIRES DE LÉVY
# ============================================================================

def calculate_levy_areas(prices: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    """
    Calcule les vraies Aires de Lévy pour toutes les paires d'actifs.
    
    Pour deux séries temporelles normalisées X et Y (représentant un mouvement 
    brownien plan), l'aire de Lévy est définie comme :
    
    A = (1/2) * ∫ (X dY - Y dX)
    
    En discret (Stratonovich, midpoint) :
    A ≈ (1/2) * Σ ((X_i + X_{i+1})/2 * ΔY_i - (Y_i + Y_{i+1})/2 * ΔX_i)
    
    Args:
        prices: DataFrame avec les prix des actifs
        
    Returns:
        Dictionnaire {(asset1, asset2): aire_levy}
    """
    levy_areas = {}
    
    # Normaliser les prix (log-returns cumulés)
    log_returns = np.log(prices / prices.iloc[0])
    
    # Calculer pour chaque paire
    assets = log_returns.columns
    
    for i, asset1 in enumerate(assets):
        for j, asset2 in enumerate(assets):
            if i < j:  # Éviter les doublons et la diagonale
                # Filtrer les NaN pour cette paire
                pair = log_returns[[asset1, asset2]].dropna()
                if len(pair) < 10:
                    levy_areas[(asset1, asset2)] = np.nan
                    levy_areas[(asset2, asset1)] = np.nan
                    continue

                X = pair[asset1].values
                Y = pair[asset2].values

                # Calcul des incréments
                dX = np.diff(X)
                dY = np.diff(Y)
                
                # Valeurs au milieu de l'intervalle (intégrale de Stratonovich)
                X_mid = (X[:-1] + X[1:]) / 2.0
                Y_mid = (Y[:-1] + Y[1:]) / 2.0
                
                # Aire de Lévy (formule discrète)
                levy_area = 0.5 * np.sum(X_mid * dY - Y_mid * dX)
                
                levy_areas[(asset1, asset2)] = levy_area
                levy_areas[(asset2, asset1)] = -levy_area  # Antisymétrie
    
    return levy_areas


def calculate_levy_areas_vs_reference(
    prices: pd.DataFrame,
    reference_asset: str = None
) -> pd.Series:
    """
    Calcule les Aires de Lévy de tous les actifs par rapport à un actif de référence.
    Optimisé : ne calcule que les paires impliquant la référence (O(N) au lieu de O(N²)).

    Args:
        prices: DataFrame avec les prix
        reference_asset: Actif de référence (si None, utilise le premier)

    Returns:
        Series avec les aires de Lévy {asset: aire}
    """
    if reference_asset is None:
        reference_asset = prices.columns[0]

    if reference_asset not in prices.columns:
        raise ValueError(f"Actif de référence {reference_asset} introuvable")

    log_returns = np.log(prices / prices.iloc[0])

    areas_vs_ref = {}
    for asset in prices.columns:
        if asset != reference_asset:
            pair = log_returns[[reference_asset, asset]].dropna()
            if len(pair) < 10:
                areas_vs_ref[asset] = np.nan
                continue

            X = pair[reference_asset].values
            Y = pair[asset].values

            dX = np.diff(X)
            dY = np.diff(Y)
            X_mid = (X[:-1] + X[1:]) / 2.0
            Y_mid = (Y[:-1] + Y[1:]) / 2.0

            areas_vs_ref[asset] = 0.5 * np.sum(X_mid * dY - Y_mid * dX)

    return pd.Series(areas_vs_ref)


def calculate_levy_area_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la matrice complète des Aires de Lévy.
    
    Returns:
        DataFrame matrice antisymétrique des aires de Lévy
    """
    levy_areas = calculate_levy_areas(prices)
    
    assets = prices.columns
    n = len(assets)
    
    matrix = np.zeros((n, n))
    
    for i, asset1 in enumerate(assets):
        for j, asset2 in enumerate(assets):
            if i != j:
                matrix[i, j] = levy_areas.get((asset1, asset2), 0)
    
    return pd.DataFrame(matrix, index=assets, columns=assets)


def get_levy_path_data(
    prices: pd.DataFrame,
    asset_a: str,
    asset_b: str
) -> Dict[str, Any]:
    """
    Prépare les chemins normalisés et l'aire de Lévy pour une paire d'actifs.
    Utilisé pour la visualisation paramétrique.

    Args:
        prices: DataFrame avec les prix
        asset_a: Premier actif (axe X)
        asset_b: Second actif (axe Y)

    Returns:
        Dictionnaire avec clés: X, Y, dates, levy_area, asset_a, asset_b
    """
    if asset_a not in prices.columns:
        raise ValueError(f"Actif {asset_a} introuvable dans les données")
    if asset_b not in prices.columns:
        raise ValueError(f"Actif {asset_b} introuvable dans les données")

    # Log-returns normalisés, filtrer NaN
    pair_prices = prices[[asset_a, asset_b]].dropna()
    log_returns = np.log(pair_prices / pair_prices.iloc[0])

    X = log_returns[asset_a].values
    Y = log_returns[asset_b].values

    # Aire de Lévy (Stratonovich midpoint)
    dX = np.diff(X)
    dY = np.diff(Y)
    X_mid = (X[:-1] + X[1:]) / 2.0
    Y_mid = (Y[:-1] + Y[1:]) / 2.0
    levy_area = 0.5 * np.sum(X_mid * dY - Y_mid * dX)

    return {
        'X': X,
        'Y': Y,
        'dates': pair_prices.index,
        'levy_area': levy_area,
        'asset_a': asset_a,
        'asset_b': asset_b
    }
