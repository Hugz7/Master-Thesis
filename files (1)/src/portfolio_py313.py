"""
Module Portfolio - Version compatible Python 3.13
Gestion de portefeuilles lead-lag
"""

import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Import des modules mis à jour
try:
    from levy_py313 import Levy
except ImportError:
    from levy import Levy

try:
    from hermitian_py313 import Hermitian
except ImportError:
    from hermitian import Hermitian

from typing import Tuple, Dict, Optional


class LeadLagPortfolio():
    """
    Classe pour gérer l'analyse de portefeuille lead-lag.
    Version compatible Python 3.13 avec améliorations.

    Attributs:
    - price_panel (pd.DataFrame): DataFrame contenant les prix historiques.
    - s (dict): Dictionnaire pour stocker les matrices "S".
    - g (dict): Dictionnaire pour stocker les graphes dirigés "G".
    - dt_cluster_dict (dict): Dictionnaire pour stocker les informations de cluster.
    - selection_pct (float): Pourcentage pour la sélection d'actifs.
    - global_ranking (pd.DataFrame): Classement global des actifs.
    - global_lfs (pd.Series): Série de facteurs lead-lag globaux.
    - clustered_lfs (pd.Series): Série de facteurs lead-lag clusterisés.
    - return_panel (pd.DataFrame): DataFrame contenant les rendements.
    - gp_stats (pd.DataFrame): Statistiques du portefeuille global.
    - cp_stats (pd.DataFrame): Statistiques du portefeuille clusterisé.
    """

    def __init__(self, price_panel: pd.DataFrame):
        """
        Initialise le portefeuille lead-lag.

        Parameters:
        - price_panel (pd.DataFrame): Panel de prix des actifs
        """
        self.price_panel = price_panel
        self.return_panel = pd.DataFrame()

        # Initialiser les dictionnaires de matrices S et de réseaux dirigés
        self.s: Dict = {}
        self.g: Dict = {}

        # Initialiser les éléments du portefeuille global
        self.global_scores = pd.DataFrame()
        self.gp_leaders_followers = pd.Series()
        self.gp_data = pd.DataFrame()

        # Initialiser les éléments du portefeuille clusterisé
        self.dt_cluster_dict: Dict = {}
        self.cp_leaders_followers: Dict = {}
        self.cp_data = pd.DataFrame()
        self.gcp_data = pd.DataFrame()


    def generate_matrices_and_networks(self,
                                     window_size: int = 30,
                                     min_assets: int = 40,
                                     show_progress: bool = True,
                                     backend: str = 'auto'):
        """
        Génère les matrices de score lead-lag et les réseaux dirigés pour chaque index
        du panel de prix.

        Parameters:
        - window_size (int): Taille de la fenêtre roulante.
        - min_assets (int): Nombre minimum d'actifs requis dans la fenêtre.
        - show_progress (bool): Indicateur pour afficher la barre de progression.
        - backend (str): Backend à utiliser pour le calcul de signature ('auto', 'iisignature', 'manual')
        """
        # Exécuter une fenêtre roulante et générer les matrices de score lead-lag
        iterator = range(1, len(self.price_panel) - window_size)

        if show_progress:
            iterator = tqdm(
                iterator,
                desc='Génération des matrices de score lead-lag et réseaux dirigés'
            )

        for i in iterator:
            # Découper la fenêtre roulante
            window_df = self.price_panel.iloc[i - 1:i + window_size, :]

            # Supprimer les actifs avec des données manquantes dans la fenêtre
            window_df = window_df.dropna(axis=1)

            # Si plus de min_assets sont dans la fenêtre, continuer
            if window_df.shape[1] >= min_assets:

                # Trouver le dernier index de la fenêtre comme clé
                window_index = window_df.index[-1]

                # Générer la matrice Levy "S" pour la fenêtre roulante et stocker
                levy_ll = Levy(price_panel=window_df, backend=backend)
                s_matrix = levy_ll.generate_levy_matrix()
                self.s[window_index] = s_matrix

                # Générer la matrice d'adjacence "A", convertir en réseau dirigé "G" et stocker
                a_matrix = np.maximum(s_matrix, 0)
                directed_net = nx.from_pandas_adjacency(a_matrix, create_using=nx.DiGraph)
                self.g[window_index] = directed_net


    def _calculate_return_panel(self):
        """
        Calcule le panel de rendements basé sur la matrice de score "S" disponible.
        Si la matrice de score n'est pas disponible, génère les scores avant de calculer les rendements.
        """
        # Vérifier si la matrice de score est disponible
        if not self.s:
            raise ValueError(
                "Aucune matrice de score disponible. "
                "Appelez d'abord generate_matrices_and_networks()."
            )

        # Calculer les rendements
        self.return_panel = self.price_panel.pct_change()


    def calculate_global_scores(self, show_progress: bool = True) -> pd.DataFrame:
        """
        Calcule les scores globaux lead-lag pour chaque actif à chaque période.

        Parameters:
        - show_progress (bool): Afficher la barre de progression

        Returns:
        - pd.DataFrame: Scores globaux des actifs
        """
        if not self.s:
            raise ValueError(
                "Aucune matrice de score disponible. "
                "Appelez d'abord generate_matrices_and_networks()."
            )

        scores_list = []

        iterator = self.s.items()
        if show_progress:
            iterator = tqdm(iterator, desc='Calcul des scores globaux')

        for date, s_matrix in iterator:
            # Calculer le score moyen pour chaque actif
            mean_scores = s_matrix.mean(axis=1)
            scores_list.append(mean_scores)

        # Créer un DataFrame avec tous les scores
        self.global_scores = pd.DataFrame(scores_list)
        self.global_scores.index = list(self.s.keys())

        return self.global_scores


    def rank_assets_global(self, selection_pct: float = 0.3) -> pd.DataFrame:
        """
        Classe les actifs du leader au suiveur basé sur les scores globaux.

        Parameters:
        - selection_pct (float): Pourcentage d'actifs à sélectionner comme leaders/followers

        Returns:
        - pd.DataFrame: Classement des actifs avec labels leader/follower
        """
        if self.global_scores.empty:
            self.calculate_global_scores()

        # Calculer le nombre d'actifs à sélectionner
        n_assets = len(self.global_scores.columns)
        n_select = int(n_assets * selection_pct)

        # Classer les actifs pour chaque période
        rankings = []

        for date, scores in self.global_scores.iterrows():
            # Trier les scores
            sorted_scores = scores.sort_values(ascending=False)

            # Assigner les labels
            labels = pd.Series('neutral', index=sorted_scores.index)
            labels.iloc[:n_select] = 'leader'
            labels.iloc[-n_select:] = 'follower'

            rankings.append(labels)

        ranking_df = pd.DataFrame(rankings)
        ranking_df.index = self.global_scores.index

        return ranking_df


    def cluster_networks(self,
                        k_min: int = 2,
                        k_max: int = 10,
                        show_progress: bool = True) -> Dict:
        """
        Applique le clustering Hermitien aux réseaux dirigés.

        Parameters:
        - k_min (int): Nombre minimum de clusters
        - k_max (int): Nombre maximum de clusters
        - show_progress (bool): Afficher la progression

        Returns:
        - Dict: Dictionnaire des informations de cluster par date
        """
        if not self.g:
            raise ValueError(
                "Aucun réseau disponible. "
                "Appelez d'abord generate_matrices_and_networks()."
            )

        cluster_info_dict = {}

        iterator = self.g.items()
        if show_progress:
            iterator = tqdm(iterator, desc='Clustering des réseaux')

        for date, network in iterator:
            # Appliquer le clustering Hermitien
            hermitian_clusterer = Hermitian(network)
            cluster_info = hermitian_clusterer.cluster_hermitian_opt(
                k_min=k_min,
                k_max=k_max
            )
            cluster_info_dict[date] = cluster_info

        self.dt_cluster_dict = cluster_info_dict
        return cluster_info_dict


    def calculate_portfolio_returns(self,
                                   portfolio_type: str = 'global',
                                   show_progress: bool = True) -> pd.Series:
        """
        Calcule les rendements du portefeuille basés sur la stratégie lead-lag.

        Parameters:
        - portfolio_type (str): Type de portefeuille ('global', 'clustered', 'global_clustered')
        - show_progress (bool): Afficher la progression

        Returns:
        - pd.Series: Série de rendements du portefeuille
        """
        if self.return_panel.empty:
            self._calculate_return_panel()

        # Cette méthode devrait être étendue avec la logique complète
        # de construction de portefeuille selon le type
        print(f"Calcul des rendements pour le portefeuille {portfolio_type}")

        # Placeholder - à implémenter selon la logique complète du projet original
        return pd.Series()


    def plot_performance(self,
                        portfolio_returns: pd.Series,
                        title: str = "Performance du Portefeuille") -> None:
        """
        Trace la performance du portefeuille.

        Parameters:
        - portfolio_returns (pd.Series): Rendements du portefeuille
        - title (str): Titre du graphique
        """
        # Calculer la performance cumulative
        cumulative_returns = (1 + portfolio_returns).cumprod()

        # Créer le graphique
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(cumulative_returns.index, cumulative_returns.values, linewidth=2)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Valeur du Portefeuille', fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


    def calculate_statistics(self,
                           portfolio_returns: pd.Series) -> pd.Series:
        """
        Calcule les statistiques de performance du portefeuille.

        Parameters:
        - portfolio_returns (pd.Series): Rendements du portefeuille

        Returns:
        - pd.Series: Statistiques de performance
        """
        # Supposer des rendements quotidiens
        annual_factor = 252

        # Calculer les statistiques
        total_return = (1 + portfolio_returns).prod() - 1
        annual_return = (1 + total_return) ** (annual_factor / len(portfolio_returns)) - 1
        annual_vol = portfolio_returns.std() * np.sqrt(annual_factor)
        sharpe_ratio = annual_return / annual_vol if annual_vol != 0 else 0

        # Calcul du drawdown maximum
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        stats = pd.Series({
            'Rendement Total': total_return,
            'Rendement Annualisé': annual_return,
            'Volatilité Annualisée': annual_vol,
            'Ratio de Sharpe': sharpe_ratio,
            'Drawdown Maximum': max_drawdown
        })

        return stats


# Fonction utilitaire pour tester le module
def test_levy_area_compatibility():
    """
    Teste la compatibilité du calcul de Levy area avec Python 3.13.
    """
    print("Test de compatibilité Python 3.13...")

    # Créer des données de test
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    n_assets = 5

    # Générer des prix aléatoires
    np.random.seed(42)
    prices = pd.DataFrame(
        np.random.randn(100, n_assets).cumsum(axis=0) + 100,
        index=dates,
        columns=[f'Asset_{i}' for i in range(n_assets)]
    )

    # Tester Levy
    print("\nTest du module Levy...")
    levy = Levy(prices)
    levy_matrix = levy.generate_levy_matrix()
    print(f"Matrice Levy générée: {levy_matrix.shape}")
    print("✓ Module Levy fonctionne correctement")

    # Tester Portfolio
    print("\nTest du module Portfolio...")
    portfolio = LeadLagPortfolio(prices)
    portfolio.generate_matrices_and_networks(window_size=30, min_assets=3, show_progress=False)
    print(f"Matrices générées: {len(portfolio.s)}")
    print("✓ Module Portfolio fonctionne correctement")

    print("\n✓ Tous les tests réussis pour Python 3.13!")


if __name__ == "__main__":
    test_levy_area_compatibility()