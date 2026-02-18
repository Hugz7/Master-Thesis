"""
Module Levy - Version compatible Python 3.13
Alternative à iisignature pour le calcul de la Levy area

Deux implémentations sont fournies:
1. Implémentation manuelle de la signature (niveau 2)
2. Support optionnel pour esig si disponible
"""

import pandas as pd
import numpy as np
from typing import Union

# Essayer d'importer iisignature, sinon utiliser l'implémentation manuelle
try:
    from iisignature import sig
    USE_IISIGNATURE = True
    print("iisignature détecté - utilisation de la bibliothèque native")
except ImportError:
    USE_IISIGNATURE = False
    print("iisignature non disponible - utilisation de l'implémentation manuelle")
    
    # Essayer esig comme alternative
    try:
        import esig
        USE_ESIG = True
        print("esig détecté comme alternative")
    except ImportError:
        USE_ESIG = False
        print("Utilisation de l'implémentation pure Python")


def compute_signature_level2_manual(path: np.ndarray) -> np.ndarray:
    """
    Calcule la signature d'un chemin jusqu'au niveau 2 (implémentation manuelle).
    
    Pour un chemin 2D, la signature de niveau 2 contient:
    S = {1, S^(1), S^(2), S^(1,1), S^(1,2), S^(2,1), S^(2,2)}
    
    Parameters:
    - path: array de forme (n_points, 2) représentant le chemin
    
    Returns:
    - array contenant [1, S^(1), S^(2), S^(1,1), S^(1,2), S^(2,1), S^(2,2)]
    """
    n = len(path)
    
    # Initialisation
    # S^(0) = 1
    sig_0 = 1.0
    
    # S^(1) et S^(2) - intégrales de premier niveau
    sig_1 = 0.0
    sig_2 = 0.0
    
    # S^(1,1), S^(1,2), S^(2,1), S^(2,2) - intégrales de second niveau
    sig_11 = 0.0
    sig_12 = 0.0
    sig_21 = 0.0
    sig_22 = 0.0
    
    # Calcul itératif de la signature
    for i in range(1, n):
        # Incrément du chemin
        dx1 = path[i, 0] - path[i-1, 0]
        dx2 = path[i, 1] - path[i-1, 1]
        
        # Mise à jour des intégrales de second niveau
        # S^(i,j) += S^(i) * dx_j + 0.5 * dx_i * dx_j
        sig_11 += sig_1 * dx1 + 0.5 * dx1 * dx1
        sig_12 += sig_1 * dx2 + 0.5 * dx1 * dx2
        sig_21 += sig_2 * dx1 + 0.5 * dx2 * dx1
        sig_22 += sig_2 * dx2 + 0.5 * dx2 * dx2
        
        # Mise à jour des intégrales de premier niveau
        sig_1 += dx1
        sig_2 += dx2
    
    return np.array([sig_0, sig_1, sig_2, sig_11, sig_12, sig_21, sig_22])


class Levy():
    """
    Classe pour calculer les scores lead-lag de Levy basés sur un panel de prix d'actifs.
    Version compatible Python 3.13 avec support pour multiples backends.
    """

    def __init__(self, price_panel: pd.DataFrame, backend: str = 'auto'):
        """
        Initialise Levy avec un panel de prix d'actifs, calcule les rendements et les standardise.

        Parameters:
        - price_panel (pd.DataFrame): DataFrame contenant les données de prix des actifs.
        - backend (str): Backend à utiliser ('auto', 'iisignature', 'esig', 'manual')
        """
        if not isinstance(price_panel, pd.DataFrame):
            raise ValueError("L'entrée doit être un DataFrame.")

        self.data = self._preprocess_data(price_panel)
        self._set_backend(backend)
        

    def _set_backend(self, backend: str):
        """Définit le backend à utiliser pour le calcul de signature."""
        if backend == 'auto':
            if USE_IISIGNATURE:
                self.backend = 'iisignature'
            elif USE_ESIG:
                self.backend = 'esig'
            else:
                self.backend = 'manual'
        else:
            self.backend = backend
            
        print(f"Backend utilisé: {self.backend}")
        

    def _preprocess_data(self, price_panel: pd.DataFrame) -> pd.DataFrame:
        """
        Prétraite les données en calculant les rendements et en les standardisant.

        Parameters:
        - price_panel (pd.DataFrame): DataFrame contenant les données de prix.

        Returns:
        - pd.DataFrame: DataFrame traité avec rendements standardisés.
        """
        returns = price_panel.pct_change().dropna()
        standardized_returns = (returns - returns.mean()) / returns.std()
        return standardized_returns
    

    def calc_levy_area(self, path: np.ndarray) -> float:
        """
        Calcule l'aire de Lévy basée sur le chemin fourni.
        
        Utilise le backend configuré (iisignature, esig ou implémentation manuelle).

        Parameters:
        - path (np.ndarray): Array représentant le chemin (n_points, 2).

        Returns:
        - float: Aire de Lévy.
        """
        if self.backend == 'iisignature':
            path_sig = sig(path, 2)
            levy_area = 0.5 * (path_sig[4] - path_sig[5])
            
        elif self.backend == 'esig':
            # esig utilise une interface différente
            import esig
            path_sig = esig.stream2sig(path, 2)
            levy_area = 0.5 * (path_sig[4] - path_sig[5])
            
        else:  # backend == 'manual'
            path_sig = compute_signature_level2_manual(path)
            # Indices: [0]=1, [1]=S^(1), [2]=S^(2), [3]=S^(1,1), [4]=S^(1,2), [5]=S^(2,1), [6]=S^(2,2)
            levy_area = 0.5 * (path_sig[4] - path_sig[5])
        
        return levy_area
    

    def generate_levy_matrix(self) -> pd.DataFrame:
        """
        Génère la matrice de score lead-lag de Levy pour le panel de prix donné.

        Returns:
        - pd.DataFrame: Matrice de score lead-lag de Levy.
        """
        assets = self.data.columns
        n = len(assets)
        levy_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i+1, n):
                pair_path = self.data[[assets[i], assets[j]]].values
                val = self.calc_levy_area(pair_path)
                levy_matrix[i, j] = val
                levy_matrix[j, i] = -val

        levy_matrix_df = pd.DataFrame(levy_matrix, index=assets, columns=assets)
        return levy_matrix_df
    

    def score_assets(self) -> pd.DataFrame:
        """
        Calcule la moyenne de chaque ligne dans la matrice de Levy comme score de l'actif correspondant.

        Returns:
        - pd.DataFrame: DataFrame avec les scores des actifs.
        """
        levy_matrix = self.generate_levy_matrix()
        lead_lag_score = levy_matrix.mean(axis=1)
        lead_lag_df = pd.DataFrame({self.data.index[-1]: lead_lag_score})
        return lead_lag_df.T
