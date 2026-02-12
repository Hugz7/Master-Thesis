"""
Module de génération de stratégies de trading pour l'application Lead-Lag Analysis
"""

import pandas as pd
import numpy as np
from typing import Dict


def generate_trading_strategy(
    mean_scores: pd.Series,
    crypto_prices: pd.DataFrame,
    traditional_prices: pd.DataFrame,
    scores: pd.DataFrame,
    window_size: int
) -> Dict[str, any]:
    """
    Génère une stratégie de trading HFT basée sur l'analyse lead-lag
    """
    strategy = {
        'signal': None,
        'confidence': None,
        'leaders': [],
        'followers': [],
        'recommendations': [],
        'risks': [],
        'opportunities': []
    }
    
    # Identifier cryptos et tradis
    crypto_leaders = mean_scores[[c for c in crypto_prices.columns if c in mean_scores.index and mean_scores[c] > 0]]
    crypto_followers = mean_scores[[c for c in crypto_prices.columns if c in mean_scores.index and mean_scores[c] < 0]]
    trad_leaders = mean_scores[[t for t in traditional_prices.columns if t in mean_scores.index and mean_scores[t] > 0]]
    
    strategy['leaders'] = mean_scores.nlargest(3).to_dict()
    strategy['followers'] = mean_scores.nsmallest(3).to_dict()
    
    # Volatilité des scores (stabilité)
    score_volatility = scores.std()
    
    # Signal principal
    if len(crypto_leaders) > len(crypto_followers):
        strategy['signal'] = 'CRYPTO_LEAD'
        strategy['confidence'] = 'HAUTE' if mean_scores[crypto_leaders.index].mean() > 0.3 else 'MOYENNE'
    elif len(trad_leaders) > 2 and mean_scores[trad_leaders.index].mean() > 0.2:
        strategy['signal'] = 'MARKET_LEAD'
        strategy['confidence'] = 'HAUTE'
    else:
        strategy['signal'] = 'MIXED'
        strategy['confidence'] = 'FAIBLE'
    
    # Recommandations selon le signal
    if strategy['signal'] == 'CRYPTO_LEAD':
        top_leader = crypto_leaders.idxmax()
        top_follower = crypto_followers.idxmin() if len(crypto_followers) > 0 else None
        lag_estimate = "1-3 heures" if window_size < 20 else "4-12 heures"
        
        strategy['recommendations'] = [
            f"🎯 **Setup HFT Principal**: Monitorer {top_leader} en temps réel (tick-by-tick)",
            f"⚡ **Trigger d'entrée**: Dès que {top_leader} monte de +0.5-1%, acheter immédiatement les followers crypto",
            f"📊 **Actifs à trader**: {', '.join(crypto_followers.nlargest(3).index.tolist())} (délai estimé: {lag_estimate})",
            f"💰 **Take-profit agressif**: +0.3-0.8% (scalping, sortir rapidement)",
            f"🛡️ **Stop-loss serré**: -0.2% (HFT = gestion stricte du risque)",
            f"⏱️ **Holding time**: 15 min - 4h maximum selon la volatilité"
        ]
        strategy['opportunities'] = [
            f"Signal précoce: {top_leader} précède le marché de {lag_estimate}",
            f"Arbitrage statistique: Corrélation {mean_scores[top_leader]:.3f} avec les followers",
            "Utiliser des ordres limites pour éviter le slippage",
            "Exploiter le momentum pendant la fenêtre de lag"
        ]
        strategy['risks'] = [
            f"⚠️ Faux signal si {top_leader} fait un spike isolé (vérifier volume > moyenne 20j)",
            "⚠️ Slippage élevé sur cryptos peu liquides (préférer les Top 10)",
            "⚠️ Retournement brutal si news crypto négatives (stop-loss obligatoire)",
            f"⚠️ Le lag peut varier: tester sur données historiques avant de trader réel"
        ]
    
    elif strategy['signal'] == 'MARKET_LEAD':
        top_trad_leader = trad_leaders.idxmax()
        lag_estimate = "2-6 heures" if window_size < 30 else "6-24 heures"
        
        strategy['recommendations'] = [
            f"🎯 **Setup HFT**: Surveiller {top_trad_leader} (ouverture marchés US 15h30 CET crucial)",
            f"⚡ **Trigger**: Si {top_trad_leader} +0.5% → acheter BTC/ETH dans les 15-30 min",
            f"📊 **Paires à trader**: BTC-USD, ETH-USD (plus liquides, moins de slippage)",
            f"💰 **Target profit**: +0.4-1% (les cryptos amplifient souvent le mouvement)",
            f"🛡️ **Stop-loss**: -0.3% (protection contre décorrélation soudaine)",
            f"⏱️ **Window**: Positions de 1-6h, surveiller clôture marchés US (22h CET)"
        ]
        strategy['opportunities'] = [
            f"{top_trad_leader} donne un signal {lag_estimate} avant les cryptos",
            "Profiter de l'ouverture/clôture des marchés traditionnels (gaps)",
            "Corréler avec VIX: forte volatilité tradis = amplification crypto",
            "Weekend: cryptos continuent, tradis fermés → opportunités uniques"
        ]
        strategy['risks'] = [
            f"⚠️ Les cryptos peuvent ignorer les indices si news crypto dominantes",
            "⚠️ Weekends/jours fériés: tradis fermés, cryptos peuvent diverger",
            "⚠️ Flash crash possible sur crypto même si indices stables",
            "⚠️ Réglementation crypto peut casser la corrélation instantanément"
        ]
    
    else:  # MIXED
        strategy['recommendations'] = [
            "⚖️ **Pas de setup HFT clair** - éviter le trading directionnel",
            "📊 **Alternative**: Market-making ou arbitrage de spread",
            "⏰ **Attendre signal net**: score leader > 0.4 avant d'entrer",
            "🔍 **Analyser timeframes courts**: regarder fenêtres 5-10 jours pour signaux émergents",
            "💼 **Stratégie défensive**: DCA sur leaders historiques (BTC, ETH)",
            "📈 **Zones d'accumulation**: Profiter de l'incertitude pour acheter dips"
        ]
        strategy['opportunities'] = [
            "Phase de transition = volatilité élevée = opportunités swing (3-7j)",
            "Possibilité de mean-reversion trading (pairs trading)",
            "Accumulation progressive sur actifs sous-évalués",
            "Préparation: backtester plusieurs scénarios pour prochain signal"
        ]
        strategy['risks'] = [
            "⚠️ Volatilité imprévisible sans tendance claire",
            "⚠️ Risque de whipsaw (faux signaux alternés)",
            "⚠️ Spreads élargis en période d'incertitude (coût de trading +élevé)",
            "⚠️ Nécessite surveillance constante (fatigue décisionnelle)"
        ]
    
    # Ajouter informations sur la stabilité
    most_stable = score_volatility.nsmallest(3)
    strategy['stable_assets'] = most_stable.to_dict()
    
    return strategy
