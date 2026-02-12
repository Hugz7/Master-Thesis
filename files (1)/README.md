# 📊 Crypto Lead-Lag Analysis Pro

Application Streamlit modulaire pour analyser les relations lead-lag entre cryptomonnaies et marchés traditionnels, avec calcul des vraies **Aires de Lévy**.

## 📁 Structure du Projet

```
.
├── main.py              # 🚀 FICHIER PRINCIPAL - Lancer celui-ci
├── utils.py             # 🛠️ Fonctions utilitaires & calculs mathématiques
├── visualizations.py    # 📊 Toutes les fonctions de graphiques Plotly
├── strategy.py          # 🎯 Génération de stratégies de trading HFT
├── requirements.txt     # 📦 Dépendances Python
└── src/
    └── portfolio_py313/ # 📚 Module d'analyse lead-lag
```

## 🚀 Installation

```bash
# Installer les dépendances
pip install -r requirements.txt

# Installer NetworkX pour graphes
pip install networkx

# Lancer l'application
streamlit run main.py
```

## 📐 Aires de Lévy - Explications Mathématiques

Pour deux séries temporelles normalisées X(t) et Y(t), l'aire de Lévy est :

$$A = \frac{1}{2} \int_0^T (X_t dY_t - Y_t dX_t)$$

**Implémentation discrète** :

$$A \approx \frac{1}{2} \sum_{i=1}^{n-1} (X_i \Delta Y_i - Y_i \Delta X_i)$$

**Interprétation** :
- **A > 0** : X mène Y (X bouge avant Y)
- **A < 0** : Y mène X (Y bouge avant X)  
- **|A| grand** : Relation forte et persistante
- **A ≈ 0** : Mouvements indépendants ou synchrones

## 🎨 Correctifs Appliqués

### ✅ Texte NOIR sur Fonds Colorés

Tous les textes dans les boxes de stratégie utilisent maintenant `color: #000000 !important` :

- **Actions Concrètes** : Fond `#b3d9ff` (bleu moyen) + texte noir
- **Opportunités** : Fond `#a5d6a7` (vert moyen) + texte noir
- **Risques** : Fond `#ffe082` (jaune moyen) + texte noir
- **Leaders** : Fond `#c8e6c9` (vert clair) + texte noir
- **Followers** : Fond `#ffcdd2` (rouge clair) + texte noir

### ✅ Architecture Modulaire

Le code de 2100+ lignes est maintenant séparé en 4 modules :

1. **utils.py** (440 lignes) : Téléchargement, validation, alignement, **calcul Aires de Lévy**
2. **visualizations.py** (380 lignes) : Tous les graphiques Plotly
3. **strategy.py** (140 lignes) : Génération stratégies HFT
4. **main.py** (340 lignes) : Interface Streamlit

### ✅ Vraies Aires de Lévy

Implémentation mathématique rigoureuse :
- Normalisation par log-returns cumulés
- Intégrale de Stratonovich discrète
- Matrice antisymétrique complète
- Visualisation heatmap + graphique vs référence

## 📊 Onglets Disponibles

1. **📊 Classements** : Barres horizontales cryptos vs tradis
2. **🕸️ Réseau** : Graphe dirigé des relations lead-lag
3. **📈 Score/Volatilité** : Scatter plot risque vs influence
4. **📊 Distribution** : Histogramme des scores
5. **📐 Aires de Lévy** : Heatmap + graphique vs référence ⭐ NOUVEAU
6. **💾 Export** : CSV avec scores et classements

## 🎯 Recommandations HFT Concrètes

Les stratégies générées incluent maintenant :

- **Triggers d'entrée précis** : "Si BTC +0.5-1% → acheter ETH dans 15-30 min"
- **Timing exact** : "Ouverture US 15h30 CET", "Holding 1-6h"
- **Objectifs chiffrés** : "Take-profit +0.3-0.8%", "Stop-loss -0.2%"
- **Estimation de lag** : "1-3 heures" ou "4-12 heures" selon fenêtre

## 🔍 Cas d'Usage

### Trading Haute Fréquence
```
1. Identifier leader principal (ex: DOT score +0.58)
2. Monitorer en temps réel
3. Trigger : DOT +0.5% → Acheter ADA, SOL
4. Target : +0.4-0.8% en 15min - 4h
5. Stop : -0.2%
```

### Analyse Géométrique (Aires de Lévy)
```
1. Calculer matrice des aires de Lévy
2. Identifier paires avec |A| > 0.5
3. Aire positive = Premier actif mène
4. Utiliser pour confirmer lead-lag
```

### Arbitrage Statistique
```
1. Détecter décalage temporel (lag)
2. Calculer corrélation + aire de Lévy
3. Entrer quand leader bouge
4. Sortir quand follower rattrape
```

## ⚙️ Configuration Recommandée

### Bitcoin vs Marchés
- Cryptos : Bitcoin seul
- Tradis : SPY, QQQ, GLD
- Période : 2 ans
- Fenêtre : 30 jours

### DeFi vs Tech
- Cryptos : DeFi preset (ETH, UNI, LINK, AVAX)
- Tradis : AAPL, MSFT, GOOGL, META
- Période : 1 an
- Fenêtre : 20 jours

### Analyse Complète
- Cryptos : Top 10
- Tradis : Mix complet (SPY, QQQ, GLD, AAPL, MSFT)
- Période : 2 ans
- Fenêtre : 30 jours
- ✅ Activer Aires de Lévy

## 📝 Notes Techniques

### Gestion des Weekends
- Cryptos : 7j/7 (trading continu)
- Tradis : 5j/7 (fermés weekend)
- **Solution** : Forward fill max 2 jours
- **Résultat** : ~200+ valeurs weekend complétées

### Performance
- ~2-5 secondes : Téléchargement données
- ~3-10 secondes : Calcul lead-lag (dépend fenêtre)
- ~5-15 secondes : Calcul Aires de Lévy (optionnel)
- **Total** : < 30 secondes pour analyse complète

### Limites
- Min 20 jours de données requis
- Min 3 actifs après nettoyage
- Aires de Lévy : calcul O(n²) en nombre d'actifs

## 🐛 Dépannage

### Erreur "portfolio_py313 not found"
```bash
# Vérifier que le dossier src/ existe
ls src/portfolio_py313/

# Installer le module
cd src/portfolio_py313
pip install -e .
```

### Graphiques ne s'affichent pas
```bash
# Vérifier plotly
pip install --upgrade plotly

# Vérifier networkx
pip install networkx
```

### Texte illisible
- ✅ Corrigé : Tous les textes sont maintenant en noir `#000000` sur fonds colorés
- Si problème persiste : Désactiver "Mode sombre" dans options

## 📚 Références Mathématiques

**Aires de Lévy** :
- Lévy, P. (1940). "Le mouvement brownien plan"
- Formule de Stratonovich pour intégrales stochastiques
- Application en finance : Détection de causalité géométrique

**Lead-Lag Analysis** :
- Corrélations croisées temporelles
- Fenêtres glissantes
- Scores agrégés sur période

## 🎓 Auteurs

Développé pour l'analyse quantitative des marchés crypto.

**Version** : 3.0 - Modulaire + Aires de Lévy
**Date** : 2026-02-12
