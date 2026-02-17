"""
Module portfolio_py313 - Analyse Lead-Lag par aires de Levy (Stratonovich)

Calcule les scores lead-lag entre actifs financiers en utilisant
des fenetres glissantes et la formule discrete de l'aire de Levy.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class LeadLagPortfolio:
    """
    Analyse lead-lag basee sur les aires de Levy avec fenetres glissantes.

    Pour chaque fenetre temporelle, calcule la matrice antisymetrique
    des aires de Levy entre toutes les paires d'actifs, puis derive
    un score net par actif (moyenne des lignes).

    Score > 0 : l'actif est un leader (il precede les autres)
    Score < 0 : l'actif est un follower (il suit les autres)
    """

    def __init__(self, prices: pd.DataFrame):
        """
        Parameters
        ----------
        prices : pd.DataFrame
            Index = dates, colonnes = noms d'actifs, valeurs = prix.
            Peut contenir des NaN.
        """
        self.prices = prices.copy()
        self.assets = list(prices.columns)
        self.n_assets = len(self.assets)

        self._windows: List[Tuple[int, int, List[str]]] = []
        self._levy_matrices: List[np.ndarray] = []
        self._scores: Optional[pd.DataFrame] = None

    def generate_matrices_and_networks(
        self,
        window_size: int = 30,
        min_assets: int = 3,
        show_progress: bool = False,
        backend: str = 'manual'
    ) -> None:
        """
        Calcule les matrices d'aires de Levy pour chaque fenetre glissante.

        Parameters
        ----------
        window_size : int
            Taille de la fenetre en jours (10-90).
        min_assets : int
            Nombre minimum d'actifs valides par fenetre.
        show_progress : bool
            Afficher la progression (non utilise en mode Streamlit).
        backend : str
            Backend de calcul ('manual' = numpy vectorise).
        """
        prices = self.prices
        n_days = len(prices)
        assets = self.assets

        self._windows = []
        self._levy_matrices = []

        min_data_points = max(10, window_size // 2)

        for start in range(0, n_days - window_size + 1):
            end = start + window_size
            window_prices = prices.iloc[start:end]

            # Actifs avec assez de donnees non-NaN dans cette fenetre
            valid_mask = window_prices.notna().sum() >= min_data_points
            valid_cols = [col for col in assets if valid_mask[col]]

            if len(valid_cols) < min_assets:
                continue

            # Nettoyer la fenetre
            wp = window_prices[valid_cols].copy()
            wp = wp.ffill(limit=2).bfill(limit=1).dropna(how='any')

            if len(wp) < min_data_points:
                continue

            # Log-returns cumules depuis le debut de la fenetre
            first_row = wp.iloc[0]
            if (first_row == 0).any():
                continue

            log_returns = np.log(wp / first_row)
            vals = log_returns.values  # (T, n_valid)

            # Matrice d'aires de Levy vectorisee
            # A(X,Y) = 0.5 * sum( X_mid * dY - Y_mid * dX )
            dV = np.diff(vals, axis=0)                # (T-1, n_valid)
            V_mid = (vals[:-1] + vals[1:]) / 2.0      # (T-1, n_valid)

            A = V_mid.T @ dV                           # (n_valid, n_valid)
            levy_matrix = 0.5 * (A - A.T)             # antisymetrique

            self._windows.append((start, end, valid_cols))
            self._levy_matrices.append(levy_matrix)

    def calculate_global_scores(self, show_progress: bool = False) -> pd.DataFrame:
        """
        Calcule un score lead-lag par actif par fenetre.

        Returns
        -------
        pd.DataFrame
            Shape (n_fenetres, n_actifs). Colonnes = noms d'actifs.
            Valeurs dans [-1, +1]. NaN si l'actif est absent de la fenetre.
        """
        all_scores = []

        for idx, (start, end, valid_cols) in enumerate(self._windows):
            levy_matrix = self._levy_matrices[idx]
            n_valid = len(valid_cols)

            # Moyenne des lignes = influence lead-lag nette
            if n_valid > 1:
                row_means = levy_matrix.sum(axis=1) / (n_valid - 1)
            else:
                row_means = np.zeros(n_valid)

            # Normaliser dans [-1, +1]
            max_abs = np.max(np.abs(row_means))
            if max_abs > 0:
                row_means = row_means / max_abs

            # Construire la ligne pour tous les actifs
            score_row = {}
            for i, col in enumerate(valid_cols):
                score_row[col] = row_means[i]

            all_scores.append(score_row)

        self._scores = pd.DataFrame(all_scores, columns=self.assets)
        return self._scores

    def rank_assets_global(self, selection_pct: float = 0.3) -> dict:
        """
        Classe les actifs par score moyen.

        Parameters
        ----------
        selection_pct : float
            Pourcentage top/bottom pour leaders/followers.

        Returns
        -------
        dict
            Cles: 'leaders', 'followers', 'neutral', 'leader_scores', 'follower_scores'
        """
        if self._scores is None:
            raise ValueError("Appelez calculate_global_scores() d'abord")

        mean_scores = self._scores.mean().sort_values(ascending=False)
        n = len(mean_scores.dropna())
        n_select = max(1, int(n * selection_pct))

        leaders = mean_scores.head(n_select)
        followers = mean_scores.tail(n_select)

        if n > 2 * n_select:
            neutral = mean_scores.iloc[n_select:-n_select]
        else:
            neutral = pd.Series(dtype=float)

        return {
            'leaders': leaders.index.tolist(),
            'followers': followers.index.tolist(),
            'neutral': neutral.index.tolist(),
            'leader_scores': leaders.to_dict(),
            'follower_scores': followers.to_dict(),
        }
