# -*- coding: utf-8 -*-
"""
Module de Backtest Lead-Lag - Donnees Horaires
Strategie : Detecter les mouvements du leader, trader le follower
Exchange simule : Binance | Frais : 0.1% par trade
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import yfinance as yf


# ============================================================================
# CONFIGURATION
# ============================================================================

FEES = 0.001          # 0.1% Binance
INITIAL_CAPITAL = 100_000.0
POSITION_SIZE_PCT = 0.05     # 5% du portefeuille par trade
SIGNAL_THRESHOLD = 0.015     # +/- 1.5% sur 1 bougie horaire
STOP_LOSS_PCT = 0.02         # -2%
TAKE_PROFIT_PCT = 0.03       # +3%
MAX_HOLDING_HOURS = 3        # Sortie forcee apres 3h
MIN_LEVY_SCORE = 0.05        # Score de Levy minimum pour trader la paire


# ============================================================================
# STRUCTURES DE DONNEES
# ============================================================================

@dataclass
class Trade:
    pair: str
    leader: str
    follower: str
    direction: str          # 'LONG' ou 'SHORT'
    entry_time: datetime
    entry_price: float
    size_usd: float
    qty: float
    stop_loss: float
    take_profit: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None   # 'TP', 'SL', 'TIMEOUT', 'EOD'
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    fees_paid: float = 0.0


@dataclass
class BacktestResult:
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    daily_returns: pd.Series = field(default_factory=pd.Series)
    metrics: Dict = field(default_factory=dict)


# ============================================================================
# TELECHARGEMENT DONNEES HORAIRES
# ============================================================================

def download_hourly_data(
    crypto_list: List[str],
    traditional_list: List[str],
    period: str = "60d"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Telecharge les donnees horaires via yfinance.
    Retourne (crypto_prices, traditional_prices)
    """
    crypto_map = {
        'bitcoin': 'BTC-USD', 'ethereum': 'ETH-USD', 'binancecoin': 'BNB-USD',
        'solana': 'SOL-USD', 'cardano': 'ADA-USD', 'ripple': 'XRP-USD',
        'polkadot': 'DOT-USD', 'avalanche-2': 'AVAX-USD', 'chainlink': 'LINK-USD',
        'litecoin': 'LTC-USD', 'dogecoin': 'DOGE-USD', 'matic-network': 'MATIC-USD'
    }

    crypto_prices = {}
    for coin in crypto_list:
        ticker = crypto_map.get(coin, f"{coin.upper()}-USD")
        try:
            df = yf.download(ticker, period=period, interval="1h", progress=False, auto_adjust=True)
            if not df.empty:
                close = df['Close']
                if hasattr(close, 'squeeze'):
                    close = close.squeeze()
                crypto_prices[coin] = close
        except Exception:
            pass

    traditional_prices = {}
    for ticker in traditional_list:
        try:
            df = yf.download(ticker, period=period, interval="1h", progress=False, auto_adjust=True)
            if not df.empty:
                close = df['Close']
                if hasattr(close, 'squeeze'):
                    close = close.squeeze()
                traditional_prices[ticker] = close
        except Exception:
            pass

    crypto_df = pd.DataFrame(crypto_prices).ffill().dropna(how='all')
    trad_df = pd.DataFrame(traditional_prices).ffill().dropna(how='all')

    return crypto_df, trad_df


# ============================================================================
# MOTEUR DE BACKTEST
# ============================================================================

class LeadLagBacktest:
    """
    Moteur de backtest pour la strategie lead-lag horaire.
    """

    def __init__(
        self,
        crypto_prices: pd.DataFrame,
        traditional_prices: pd.DataFrame,
        levy_scores: pd.Series,
        initial_capital: float = INITIAL_CAPITAL,
        position_size_pct: float = POSITION_SIZE_PCT,
        signal_threshold: float = SIGNAL_THRESHOLD,
        stop_loss_pct: float = STOP_LOSS_PCT,
        take_profit_pct: float = TAKE_PROFIT_PCT,
        max_holding_hours: int = MAX_HOLDING_HOURS,
        min_levy_score: float = MIN_LEVY_SCORE,
        fees: float = FEES
    ):
        self.all_prices = pd.concat([crypto_prices, traditional_prices], axis=1).ffill().dropna(how='all')
        self.levy_scores = levy_scores
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.signal_threshold = signal_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_holding_hours = max_holding_hours
        self.min_levy_score = min_levy_score
        self.fees = fees
        self.trades: List[Trade] = []
        self.equity_history = []

    def _get_valid_pairs(self) -> List[Tuple[str, str]]:
        """
        Retourne les paires (leader, follower) dont le score de Levy est suffisant.
        Le leader a un score positif, le follower un score negatif.
        """
        pairs = []
        assets = self.all_prices.columns.tolist()
        sorted_scores = self.levy_scores.sort_values(ascending=False)

        leaders = sorted_scores[sorted_scores > self.min_levy_score].index.tolist()
        followers = sorted_scores[sorted_scores < -self.min_levy_score].index.tolist()

        for leader in leaders:
            for follower in followers:
                if leader != follower and leader in assets and follower in assets:
                    pairs.append((leader, follower))

        return pairs[:10]  # Max 10 paires pour eviter le surfit

    def _calc_hourly_return(self, asset: str, idx: int) -> Optional[float]:
        """Calcule le rendement sur la bougie courante."""
        if idx < 1 or asset not in self.all_prices.columns:
            return None
        prev = self.all_prices[asset].iloc[idx - 1]
        curr = self.all_prices[asset].iloc[idx]
        if prev == 0 or np.isnan(prev) or np.isnan(curr):
            return None
        return (curr - prev) / prev

    def _check_exit(self, trade: Trade, prices: pd.Series, current_time: datetime, hours_held: int) -> Optional[Tuple[float, str]]:
        """
        Verifie si le trade doit etre ferme.
        Retourne (exit_price, reason) ou None.
        """
        price = prices.get(trade.follower)
        if price is None or np.isnan(price):
            return None

        if trade.direction == 'LONG':
            pnl_pct = (price - trade.entry_price) / trade.entry_price
            if price <= trade.stop_loss:
                return (trade.stop_loss, 'SL')
            if price >= trade.take_profit:
                return (trade.take_profit, 'TP')
        else:  # SHORT
            pnl_pct = (trade.entry_price - price) / trade.entry_price
            if price >= trade.stop_loss:
                return (trade.stop_loss, 'SL')
            if price <= trade.take_profit:
                return (trade.take_profit, 'TP')

        if hours_held >= self.max_holding_hours:
            return (price, 'TIMEOUT')

        return None

    def _open_trade(self, leader: str, follower: str, direction: str,
                    entry_price: float, entry_time: datetime) -> Trade:
        """Ouvre un nouveau trade."""
        size_usd = self.capital * self.position_size_pct
        fees_entry = size_usd * self.fees
        size_usd_net = size_usd - fees_entry
        qty = size_usd_net / entry_price

        if direction == 'LONG':
            stop_loss = entry_price * (1 - self.stop_loss_pct)
            take_profit = entry_price * (1 + self.take_profit_pct)
        else:
            stop_loss = entry_price * (1 + self.stop_loss_pct)
            take_profit = entry_price * (1 - self.take_profit_pct)

        trade = Trade(
            pair=f"{leader}/{follower}",
            leader=leader,
            follower=follower,
            direction=direction,
            entry_time=entry_time,
            entry_price=entry_price,
            size_usd=size_usd,
            qty=qty,
            stop_loss=stop_loss,
            take_profit=take_profit,
            fees_paid=fees_entry
        )
        return trade

    def _close_trade(self, trade: Trade, exit_price: float, exit_time: datetime, reason: str):
        """Ferme un trade et calcule le PnL."""
        fees_exit = trade.qty * exit_price * self.fees
        trade.fees_paid += fees_exit

        if trade.direction == 'LONG':
            gross_pnl = (exit_price - trade.entry_price) * trade.qty
        else:
            gross_pnl = (trade.entry_price - exit_price) * trade.qty

        net_pnl = gross_pnl - fees_exit
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = reason
        trade.pnl = net_pnl
        trade.pnl_pct = net_pnl / trade.size_usd * 100
        self.capital += net_pnl

    def run(self, show_progress: bool = True) -> BacktestResult:
        """
        Execute le backtest sur toutes les donnees horaires.
        """
        pairs = self._get_valid_pairs()
        if not pairs:
            raise ValueError("Aucune paire valide trouvee. Verifiez les scores de Levy.")

        n = len(self.all_prices)
        open_trades: Dict[str, Trade] = {}  # follower -> trade actif
        open_since: Dict[str, int] = {}     # follower -> idx d'ouverture

        equity = []

        for i in range(1, n):
            current_time = self.all_prices.index[i]
            current_prices = self.all_prices.iloc[i]

            # 1. Verifier les trades ouverts
            to_close = []
            for follower, trade in open_trades.items():
                hours_held = i - open_since[follower]
                exit_info = self._check_exit(trade, current_prices, current_time, hours_held)
                if exit_info:
                    exit_price, reason = exit_info
                    self._close_trade(trade, exit_price, current_time, reason)
                    self.trades.append(trade)
                    to_close.append(follower)

            for follower in to_close:
                del open_trades[follower]
                del open_since[follower]

            # 2. Chercher de nouveaux signaux
            for leader, follower in pairs:
                # Pas de trade si deja ouvert sur ce follower
                if follower in open_trades:
                    continue

                leader_return = self._calc_hourly_return(leader, i)
                if leader_return is None:
                    continue

                follower_price = current_prices.get(follower)
                if follower_price is None or np.isnan(follower_price):
                    continue

                # Signal LONG : leader monte de +1.5%
                if leader_return >= self.signal_threshold:
                    trade = self._open_trade(leader, follower, 'LONG', follower_price, current_time)
                    open_trades[follower] = trade
                    open_since[follower] = i

                # Signal SHORT : leader baisse de -1.5%
                elif leader_return <= -self.signal_threshold:
                    trade = self._open_trade(leader, follower, 'SHORT', follower_price, current_time)
                    open_trades[follower] = trade
                    open_since[follower] = i

            equity.append({'time': current_time, 'equity': self.capital})

        # Fermer les trades encore ouverts a la fin
        last_time = self.all_prices.index[-1]
        last_prices = self.all_prices.iloc[-1]
        for follower, trade in open_trades.items():
            exit_price = last_prices.get(follower, trade.entry_price)
            self._close_trade(trade, exit_price, last_time, 'EOD')
            self.trades.append(trade)

        equity_df = pd.DataFrame(equity).set_index('time')['equity']

        result = BacktestResult(
            trades=self.trades,
            equity_curve=equity_df
        )
        result.metrics = self._compute_metrics(equity_df)
        return result

    def _compute_metrics(self, equity: pd.Series) -> Dict:
        """Calcule les metriques de performance."""
        if equity.empty or len(self.trades) == 0:
            return {}

        returns = equity.pct_change().dropna()
        total_return = (equity.iloc[-1] - self.initial_capital) / self.initial_capital * 100

        # Sharpe annualise (horaire -> annuel : * sqrt(8760))
        sharpe = (returns.mean() / returns.std() * np.sqrt(8760)) if returns.std() > 0 else 0

        # Sortino (downside only)
        downside = returns[returns < 0].std()
        sortino = (returns.mean() / downside * np.sqrt(8760)) if downside > 0 else 0

        # Max drawdown
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max * 100
        max_dd = drawdown.min()

        # Stats trades
        completed = [t for t in self.trades if t.pnl is not None]
        if completed:
            winners = [t for t in completed if t.pnl > 0]
            losers = [t for t in completed if t.pnl <= 0]
            win_rate = len(winners) / len(completed) * 100
            avg_win = np.mean([t.pnl for t in winners]) if winners else 0
            avg_loss = np.mean([t.pnl for t in losers]) if losers else 0
            profit_factor = abs(sum(t.pnl for t in winners) / sum(t.pnl for t in losers)) if losers and sum(t.pnl for t in losers) != 0 else 0
            total_fees = sum(t.fees_paid for t in completed)

            by_reason = {}
            for t in completed:
                by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0) + 1
        else:
            win_rate = avg_win = avg_loss = profit_factor = total_fees = 0
            by_reason = {}

        return {
            'total_return_pct': round(total_return, 2),
            'final_equity': round(equity.iloc[-1], 2),
            'sharpe_ratio': round(sharpe, 3),
            'sortino_ratio': round(sortino, 3),
            'max_drawdown_pct': round(max_dd, 2),
            'total_trades': len(completed),
            'win_rate_pct': round(win_rate, 2),
            'avg_win_usd': round(avg_win, 2),
            'avg_loss_usd': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 3),
            'total_fees_usd': round(total_fees, 2),
            'exit_reasons': by_reason
        }
