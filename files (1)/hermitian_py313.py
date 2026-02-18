"""
Module Hermitian - Version compatible Python 3.13
Clustering Hermitien pour réseaux dirigés
"""

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import math
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import networkx as nx
from typing import Tuple


class Hermitian():
    """
    Classe pour le clustering Hermitien de réseaux dirigés.
    Compatible Python 3.13 avec dépendances mises à jour.
    """
    
    def __init__(self, directed_net: nx.DiGraph):
        """
        Initialise l'objet Hermitian en utilisant un réseau dirigé.
        Trouve les matrices d'adjacence (A) et d'adjacence Hermitienne (A_tilda),
        et normalise A_tilda avec la méthode Random-Walk.

        Parameters:
        - directed_net (nx.DiGraph): Réseau dirigé de scores lead-lag.
        """
        self.directed_net = directed_net
        
        # Calculer A, A_tilda et A_tilda_rw normalisée
        self.a = self.calc_adjacency()
        self.a_tilda = self.calc_hermitian_adjacency()
        self.a_tilda_rw = self.norm_hermitian_adjacency()
    
    
    def calc_adjacency(self) -> scipy.sparse.csr_matrix:
        """
        Calcule la matrice d'adjacence sparse à partir du réseau dirigé.

        Returns:
        - scipy.sparse.csr_matrix: Matrice d'adjacence, A.
        """
        return nx.adjacency_matrix(self.directed_net)
    
    
    def calc_hermitian_adjacency(self) -> scipy.sparse.csr_matrix:
        """
        Calcule la matrice d'adjacence Hermitienne sparse, A_tilda,
        à partir de la matrice d'adjacence A.

        Returns:
        - scipy.sparse.csr_matrix: Matrice d'adjacence Hermitienne, A_tilda.
        """
        return (self.a * 1j) - (self.a.transpose() * 1j)
    
    
    def norm_hermitian_adjacency(self) -> scipy.sparse.csr_matrix:
        """
        Normalise la matrice d'adjacence Hermitienne avec la méthode
        Random Walk Normalization.

        Returns:
        - scipy.sparse.csr_matrix: Matrice d'adjacence Hermitienne normalisée, A_tilda_rw.
        """
        # Étape 1: Convertir la matrice sparse en array
        adj_mat = self.a_tilda.toarray()

        # Étape 2: Calculer la matrice de degré
        # Note: Pour Python 3.13, np.sum gère correctement les nombres complexes
        deg_mat = np.diag(np.sum(np.abs(adj_mat), axis=1))

        # Étape 3: Calculer l'inverse de la racine carrée de la matrice de degré
        # Ajouter une petite valeur pour éviter la division par zéro
        deg_mat_sqrt = np.sqrt(deg_mat + 1e-10)
        inv_sqrt_deg_mat = np.linalg.inv(deg_mat_sqrt)

        # Étape 4: Effectuer la normalisation Random Walk
        adj_mat_norm = np.dot(np.dot(inv_sqrt_deg_mat, adj_mat), inv_sqrt_deg_mat.conj().T)

        # Étape 5: Convertir en matrice sparse
        adj_mat_norm = scipy.sparse.csr_matrix(adj_mat_norm)
        
        return adj_mat_norm
        
    
    def cluster_hermitian(self,
                          k: int,
                          kmeans_init: str = 'k-means++',
                          kmeans_n_init: int = 10,
                          kmeans_random_state: int = 42,
                          add_to_network: bool = False) -> Tuple[np.ndarray, np.float64]:
        """
        Applique le clustering Hermitien à la matrice d'adjacence Hermitienne normalisée
        pour un k donné.

        Parameters:
        - k (int): Nombre de clusters.
        - kmeans_init (str): Méthode d'initialisation pour KMeans. Défaut 'k-means++'.
        - kmeans_n_init (int): Nombre d'exécutions KMeans avec différentes graines. Défaut 10.
        - kmeans_random_state (int): État aléatoire pour KMeans. Défaut 42.
        - add_to_network (bool): Si True, ajoute les labels de cluster comme attribut aux nœuds.

        Returns:
        - Tuple[np.ndarray, np.float64]: Retourne (cluster_labels, silhouette_avg).
        """
        # Trouver les vecteurs propres de la matrice d'adjacence Hermitienne
        # Pour Python 3.13, scipy.sparse.linalg.eigsh fonctionne correctement
        n_eigenvectors = int(2 * math.floor(k / 2))
        eigenval, eigenvec = scipy.sparse.linalg.eigsh(
            self.a_tilda_rw, 
            k=n_eigenvectors,
            which='LM'  # Plus grandes valeurs propres en magnitude
        )

        # Préparer les données d'entrée pour KMeans à partir des vecteurs propres
        X = np.block([[np.real(eigenvec), np.imag(eigenvec)]])

        # Appliquer le clustering KMeans
        # Note: scikit-learn >= 1.3 est compatible avec Python 3.13
        clusterer = KMeans(
            n_clusters=k,
            init=kmeans_init,
            n_init=kmeans_n_init,
            random_state=kmeans_random_state
        )
        cluster_labels = clusterer.fit_predict(X)

        # Calculer le score de Silhouette
        silhouette_avg = silhouette_score(X, cluster_labels)
        
        # Ajouter les numéros de cluster aux attributs du réseau
        if add_to_network:
            cluster_dict = dict(zip(self.directed_net.nodes(), cluster_labels))
            nx.set_node_attributes(self.directed_net, cluster_dict, 'cluster')
              
        return cluster_labels, silhouette_avg
    
    
    def get_cluster_info(self) -> dict:
        """
        Récupère les labels de cluster du graphe et génère un dictionnaire.

        Returns:
        - dict: Dictionnaire avec le numéro de cluster comme clé 
                et une liste de noms de nœuds comme valeurs.
        """
        cluster_labels = nx.get_node_attributes(self.directed_net, 'cluster')
        cluster_dict = {}
        
        for node, cluster in cluster_labels.items():
            if cluster in cluster_dict:
                cluster_dict[cluster].append(node)
            else:
                cluster_dict[cluster] = [node]
                
        return cluster_dict
    
    
    def cluster_hermitian_opt(self,
                              k_min: int,
                              k_max: int,
                              kmeans_init: str = 'k-means++',
                              kmeans_n_init: int = 10,
                              kmeans_random_state: int = 42) -> dict:
        """
        Applique le clustering Hermitien pour une plage de k entre k_min et k_max.
        Utilise le score de silhouette pour trouver le nombre optimal de clusters.

        Parameters:
        - k_min (int): Nombre minimum de clusters à considérer.
        - k_max (int): Nombre maximum de clusters à considérer.
        - kmeans_init (str): Méthode d'initialisation pour KMeans. Défaut 'k-means++'.
        - kmeans_n_init (int): Nombre d'exécutions KMeans. Défaut 10.
        - kmeans_random_state (int): État aléatoire pour KMeans. Défaut 42.

        Returns:
        - dict: Dictionnaire contenant le numéro de cluster comme clés
                et la liste des nœuds dans chaque cluster comme valeurs.
                Ajoute également les labels optimaux au réseau dirigé.
        """
        # Initialiser les résultats de clustering pour le nombre optimal de clusters (ONC)
        onc_labels = None
        onc_score = -1
        optimal_k = k_min

        # Effectuer le clustering pour la plage k_min à k_max et trouver le k optimal
        for k in range(k_min, k_max + 1):
            labels, score = self.cluster_hermitian(
                k=k,
                kmeans_init=kmeans_init,
                kmeans_n_init=kmeans_n_init,
                kmeans_random_state=kmeans_random_state
            )

            # Mettre à jour le nombre optimal de clusters et les résultats
            if score > onc_score:
                onc_score = score
                onc_labels = labels
                optimal_k = k
        
        print(f"Nombre optimal de clusters: {optimal_k} (score silhouette: {onc_score:.4f})")
        
        # Ajouter les labels de cluster au graphe
        cluster_dict = dict(zip(self.directed_net.nodes(), onc_labels))
        nx.set_node_attributes(self.directed_net, cluster_dict, 'cluster')
        
        return self.get_cluster_info()
