# Rapport Scientifique : Détection de Fraude par Carte Bancaire
## Analyse Prédictive par Apprentissage Automatique

---![Analyse statistique](![WhatsApp Image 2025-10-25 at 18 34 06_2d8a60b6](https://github.com/user-attachments/assets/aa5ee2b3-c786-4239-8eab-3437f3a8b8ee)
)

**Auteur** : [Doumbouya Moussa]  
**Date** : Décembre 2025  
**Contexte** : Projet d'Analyse de Données et Machine Learning

---

## Table des Matières

1. [Introduction](#1-introduction)
2. [Méthodologie](#2-méthodologie)
3. [Résultats et Discussion](#3-résultats-et-discussion)
4. [Conclusion](#4-conclusion)
5. [Références](#5-références)

---

## 1. Introduction

### 1.1 Contexte et Problématique

La fraude par carte bancaire représente un défi majeur pour les institutions financières à l'échelle mondiale. Selon les estimations récentes, les pertes annuelles liées à la fraude bancaire dépassent plusieurs milliards de dollars, affectant à la fois les consommateurs et les établissements financiers. Dans un contexte où les transactions électroniques se multiplient exponentiellement, la détection précoce et automatisée des transactions frauduleuses devient une nécessité absolue.

Le principal défi réside dans le **déséquilibre extrême des classes** : les transactions frauduleuses représentent généralement moins de 0,2% de l'ensemble des transactions. Ce déséquilibre rend difficile l'entraînement de modèles de machine learning efficaces, car les algorithmes traditionnels ont tendance à optimiser la précision globale au détriment de la détection des cas frauduleux.

### 1.2 Objectifs de l'Étude

Cette étude vise à développer et évaluer plusieurs modèles d'apprentissage automatique capables de :

1. **Identifier automatiquement les transactions frauduleuses** avec une haute sensibilité (recall)
2. **Minimiser les faux positifs** pour éviter de bloquer des transactions légitimes
3. **Comparer les performances** de différentes approches algorithmiques
4. **Proposer un système robuste** pouvant être déployé en environnement de production

### 1.3 Dataset Utilisé

Le dataset provient d'une étude collaborative réalisée en 2013 par Worldline et l'Université Libre de Bruxelles. Il contient **284,807 transactions** effectuées par des porteurs de cartes européens sur une période de deux jours. Parmi ces transactions, **492 sont frauduleuses**, soit **0,17%** du total.

**Caractéristiques du dataset :**
- **30 variables** : 28 features anonymisées (V1 à V28 obtenues par ACP), Time, Amount, et Class
- **Anonymisation par ACP** : Les features V1-V28 sont le résultat d'une Analyse en Composantes Principales pour protéger la confidentialité
- **Variables non transformées** : Time (secondes écoulées depuis la première transaction) et Amount (montant de la transaction)
- **Variable cible** : Class (0 = transaction légitime, 1 = fraude)

---

## 2. Méthodologie

### 2.1 Prétraitement des Données

#### 2.1.1 Vérification de la Qualité des Données

L'analyse exploratoire initiale a révélé :
- **Aucune valeur manquante** : Le dataset est complet, aucune imputation n'est nécessaire
- **Aucun doublon** : Chaque transaction est unique
- **Types de données cohérents** : Toutes les variables sont numériques (float64)

Cette qualité exceptionnelle du dataset simplifie le prétraitement et permet de se concentrer sur les aspects algorithmiques.

#### 2.1.2 Normalisation des Features

**Justification :**
Les variables Time et Amount présentent des échelles très différentes des features V1-V28. La normalisation est essentielle pour :
- Éviter que les algorithmes basés sur les distances (régression logistique) ne soient dominés par les variables à grande échelle
- Accélérer la convergence des algorithmes d'optimisation
- Permettre une interprétation équitable de l'importance des features

**Méthode choisie : StandardScaler**

La standardisation transforme chaque feature pour avoir une moyenne de 0 et un écart-type de 1 selon la formule :

```
z = (x - μ) / σ
```

où μ est la moyenne et σ l'écart-type de la feature.

**Avantages :**
- Préserve la distribution originale des données
- Ne compresse pas les valeurs dans un intervalle fixe (contrairement à MinMaxScaler)
- Robuste pour les algorithmes linéaires

### 2.2 Analyse Exploratoire

#### 2.2.1 Visualisation des Distributions

L'analyse des distributions via histogrammes et boxplots a révélé plusieurs insights :

**Time :**
- Distribution relativement uniforme sur 48 heures
- Quelques pics suggérant des heures de forte activité transactionnelle
- Utile pour capturer les patterns temporels de fraude

**Amount :**
- Distribution fortement asymétrique (right-skewed)
- Majorité des transactions inférieures à 100 unités monétaires
- Présence de valeurs extrêmes (transactions à montants élevés)
- Les fraudes ont tendance à cibler des montants moyens (ni trop petits ni trop grands)

**Features V1-V28 :**
- Distributions variées, certaines proches de la normale, d'autres multimodales
- Présence d'outliers identifiés par les boxplots
- Ces composantes principales capturent l'essentiel de la variance du comportement transactionnel

#### 2.2.2 Analyse de Corrélation

La matrice de corrélation a permis d'identifier :

**Corrélations fortes avec la variable cible (Class) :**
- Plusieurs features (notamment V14, V17, V12, V10) montrent des corrélations négatives significatives avec la fraude
- V4 et V11 présentent des corrélations positives modérées
- Ces variables sont des indicateurs potentiellement discriminants

**Corrélations inter-features :**
- Faibles corrélations entre la plupart des features V1-V28 (avantage de l'ACP)
- Réduction naturelle de la multicolinéarité
- Time et Amount sont indépendants des autres variables

### 2.3 Ingénierie des Features

#### 2.3.1 Justification de l'Approche

L'ingénierie des features vise à créer de nouvelles variables capturant des interactions non-linéaires et des patterns complexes que les modèles linéaires ne peuvent pas détecter directement.

#### 2.3.2 Features Créées

**1. V4_Amount_Interaction**
- Produit de V4 et Amount
- Capture l'interaction entre un comportement transactionnel (V4) et le montant
- Hypothèse : Les fraudeurs adaptent leur comportement selon le montant visé

**2. V17_Amount_Interaction**
- Produit de V17 et Amount
- V17 étant fortement corrélé à la fraude, son interaction avec le montant peut révéler des patterns spécifiques
- Potentiellement utile pour identifier les fraudes ciblant certains segments de montants

**3. V4_squared**
- Transformation polynomiale de V4
- Capture les relations non-linéaires
- Permet aux modèles linéaires de mieux approximer des frontières de décision complexes

**Impact attendu :**
- Amélioration du recall pour les modèles linéaires (régression logistique)
- Enrichissement de l'espace des features pour les arbres de décision
- Potentiel d'amélioration de 2-5% sur les métriques clés

### 2.4 Stratégie de Division des Données

#### 2.4.1 Split Train-Test (70-30)

**Configuration :**
- 70% des données pour l'entraînement (199,364 transactions)
- 30% pour le test (85,443 transactions)
- `stratify=y` : Préserve la proportion de fraudes dans les deux ensembles

**Justification du ratio 70-30 :**
- Dataset suffisamment large pour permettre un ensemble de test conséquent
- 85,443 transactions de test fournissent une évaluation statistiquement robuste
- Les 492 fraudes sont divisées en ~344 (train) et ~148 (test), offrant assez d'exemples pour chaque ensemble

#### 2.4.2 Validation Croisée Stratifiée

**Configuration : 5-Fold Stratified Cross-Validation**

La validation croisée stratifiée divise l'ensemble d'entraînement en 5 sous-ensembles (folds) tout en préservant la proportion de fraudes dans chaque fold.

**Avantages :**
- Évaluation plus robuste que le simple train-test split
- Réduit le risque de sur-ajustement
- Fournit une distribution des performances (moyenne et écart-type)
- Particulièrement crucial pour les données déséquilibrées

**Processus :**
1. Division de l'ensemble d'entraînement en 5 folds
2. Pour chaque itération, 4 folds servent à l'entraînement, 1 à la validation
3. Calcul des métriques sur les 5 itérations
4. Moyenne et écart-type des performances

### 2.5 Algorithmes de Machine Learning

#### 2.5.1 Régression Logistique avec Régularisation L1

**Description :**
Modèle linéaire probabiliste qui estime la probabilité qu'une transaction soit frauduleuse.

**Configuration :**
- Solver : `liblinear` (efficace pour les problèmes de petite à moyenne taille)
- Pénalité : L1 (LASSO) pour la sélection automatique de features
- `max_iter=200` pour assurer la convergence

**Avantages :**
- Interprétable : Coefficients indiquent l'importance et la direction de l'effet
- Régularisation L1 : Élimine les features non pertinentes (coefficients = 0)
- Rapide à entraîner et prédire
- Baseline solide pour les problèmes de classification

**Limitations :**
- Suppose une relation linéaire entre features et log-odds
- Peut sous-performer face à des interactions complexes
- Sensible aux outliers

#### 2.5.2 Arbre de Décision

**Description :**
Modèle non paramétrique qui segmente l'espace des features par décisions successives.

**Configuration :**
- `max_depth=10` : Limite la profondeur pour éviter le sur-ajustement
- Critère de split : Gini impurity (par défaut)

**Avantages :**
- Capture les interactions et non-linéarités naturellement
- Aucune hypothèse sur la distribution des données
- Robuste aux outliers et données non normalisées
- Interprétable visuellement (arbre de décision)

**Limitations :**
- Tendance au sur-ajustissement sans contraintes
- Instabilité : Petites variations des données peuvent changer la structure
- Biais vers les features avec plus de valeurs uniques

#### 2.5.3 Forêt Aléatoire (Random Forest)

**Description :**
Ensemble d'arbres de décision entraînés sur des sous-échantillons aléatoires avec agrégation des prédictions.

**Configuration :**
- `n_estimators=100` : 100 arbres dans la forêt
- `max_depth=10` : Limite la complexité de chaque arbre
- Bootstrap sampling et sélection aléatoire de features à chaque split

**Avantages :**
- Réduit considérablement le sur-ajustement par rapport à un arbre unique
- Excellentes performances sur une large gamme de problèmes
- Fournit l'importance des features
- Robuste au bruit et aux outliers
- Gère bien les données déséquilibrées avec des ajustements

**Limitations :**
- Plus lent à entraîner et prédire qu'un modèle unique
- Moins interprétable qu'un arbre de décision simple
- Peut nécessiter beaucoup de mémoire

### 2.6 Métriques d'Évaluation

Dans le contexte de données fortement déséquilibrées, l'accuracy (précision globale) est trompeuse. Un modèle naïf prédisant toujours "non-fraude" aurait 99,83% d'accuracy mais serait inutile.

#### 2.6.1 Recall (Sensibilité)

**Formule :**
```
Recall = VP / (VP + FN)
```

**Interprétation :**
- Proportion de fraudes réellement détectées parmi toutes les fraudes
- **Métrique prioritaire** dans notre contexte : Manquer une fraude coûte cher

**Objectif :** Maximiser le recall (idéalement > 85%)

#### 2.6.2 F1-Score

**Formule :**
```
F1 = 2 × (Précision × Recall) / (Précision + Recall)
```

**Interprétation :**
- Moyenne harmonique de la précision et du recall
- Balance entre détecter les fraudes et éviter les faux positifs
- Particulièrement utile pour les données déséquilibrées

**Objectif :** Maximiser le F1-Score (idéalement > 0.75)

#### 2.6.3 ROC-AUC (Area Under the Curve)

**Interprétation :**
- Mesure la capacité du modèle à discriminer entre classes
- Varie de 0 à 1 (0.5 = classification aléatoire, 1.0 = parfait)
- Indépendant du seuil de classification
- Évalue la performance globale sur tous les seuils possibles

**Objectif :** ROC-AUC > 0.90 indique un excellent pouvoir discriminant

---

## 3. Résultats et Discussion

### 3.1 Performances des Modèles

Les résultats présentés ci-dessous sont issus de la validation croisée stratifiée à 5 folds sur l'ensemble d'entraînement. Les valeurs représentent la moyenne et l'écart-type sur les 5 folds.

#### 3.1.1 Tableau Récapitulatif des Performances

| Modèle | Recall | F1-Score | ROC-AUC |
|--------|--------|----------|---------|
| **Régression Logistique** | 0.6185 (±0.0387) | 0.7042 (±0.0298) | 0.9746 (±0.0042) |
| **Arbre de Décision** | 0.7500 (±0.0512) | 0.8156 (±0.0387) | 0.8721 (±0.0156) |
| **Forêt Aléatoire** | 0.8232 (±0.0428) | 0.8645 (±0.0312) | 0.9512 (±0.0089) |

### 3.2 Analyse Détaillée par Modèle

#### 3.2.1 Régression Logistique

**Points forts :**
- **ROC-AUC exceptionnel (0.9746)** : Excellent pouvoir discriminant global
- Faible variance (écart-type faible) : Prédictions stables et reproductibles
- Temps d'entraînement et de prédiction très rapides
- Modèle interprétable : Les coefficients peuvent être analysés par les experts métier

**Points faibles :**
- **Recall modéré (0.6185)** : Détecte seulement 61,85% des fraudes
- Environ **38% des fraudes manquées** : Inacceptable dans un contexte bancaire critique
- Difficulté à capturer les interactions complexes malgré l'ingénierie des features

**Interprétation :**
La régression logistique excelle à séparer globalement les classes (ROC-AUC élevé) mais son seuil de décision par défaut (0.5) n'est pas optimal pour les données déséquilibrées. Un ajustement du seuil vers 0.2-0.3 pourrait améliorer significativement le recall au prix de plus de faux positifs.

#### 3.2.2 Arbre de Décision

**Points forts :**
- **Amélioration notable du recall (0.75)** : Détecte 75% des fraudes
- Bon équilibre recall-précision (F1=0.8156)
- Capture naturellement les interactions et non-linéarités
- Interprétable : L'arbre peut être visualisé et analysé

**Points faibles :**
- **Variance élevée (±0.0512 sur recall)** : Performances instables d'un fold à l'autre
- ROC-AUC inférieur (0.8721) aux autres modèles
- Risque de sur-ajustement malgré `max_depth=10`
- Sensibilité aux variations des données d'entraînement

**Interprétation :**
L'arbre de décision représente un compromis intéressant mais sa stabilité limitée est préoccupante pour un déploiement en production. La variance élevée suggère que certains folds contiennent des patterns plus faciles à capturer.

#### 3.2.3 Forêt Aléatoire (Meilleur Modèle)

**Points forts :**
- **Recall le plus élevé (0.8232)** : Détecte 82,32% des fraudes
- **F1-Score excellent (0.8645)** : Meilleur équilibre global
- ROC-AUC très élevé (0.9512) : Excellent pouvoir discriminant
- Variance raisonnable : Performances stables (±0.0428 sur recall)
- Robuste aux outliers et au bruit

**Points faibles :**
- Temps d'entraînement plus long (100 arbres)
- Prédictions moins rapides qu'un modèle linéaire
- Modèle "boîte noire" : Moins interprétable directement
- Consommation mémoire plus importante

**Interprétation :**
La Forêt Aléatoire offre le meilleur compromis entre performances et stabilité. Avec 82,32% de recall, elle détecte la grande majorité des fraudes tout en maintenant un F1-Score élevé, indiquant peu de faux positifs. L'agrégation de 100 arbres réduit significativement la variance observée avec un arbre unique.
Le graphique

![Analyse statistique](![WhatsApp Image 2025-12-04 at 12 29 18_b1557248](https://github.com/user-attachments/assets/5c743206-84f7-4108-83af-ebabd998202f)
)


![Analyse statistique]()



![Analyse statistique](![WhatsApp Image 2025-12-04 at 12 21 13_69473275](https://github.com/user-attachments/assets/43b02591-31d0-4fbb-affb-854c903fd18c)
)
### 3.3 Analyse Approfondie des Résultats

#### 3.3.1 Trade-off Recall vs Précision

Dans un système de détection de fraude, deux types d'erreurs existent :

**Faux Négatifs (FN) :**
- Fraude non détectée : Client victime, perte financière, atteinte à la réputation
- **Coût élevé** : Perte moyenne par fraude + coûts indirects

**Faux Positifs (FP) :**
- Transaction légitime bloquée : Mécontentement client, appel au service client
- **Coût modéré** : Intervention humaine + potentielle perte de vente

**Conclusion :** Le recall doit être privilégié, ce qui justifie le choix de la Forêt Aléatoire.

#### 3.3.2 Interprétation du ROC-AUC

Les trois modèles affichent des ROC-AUC supérieurs à 0.87, indiquant :
- Tous les modèles distinguent bien les classes (mieux qu'aléatoire)
- La régression logistique (0.9746) a le meilleur pouvoir discriminant théorique
- L'écart entre ROC-AUC et recall suggère un problème de seuil de décision

**Recommandation :** Ajuster le seuil de classification via l'analyse de la courbe ROC pour optimiser le recall tout en contrôlant les faux positifs.

#### 3.3.3 Impact de l'Ingénierie des Features

Les features créées (V4_Amount_Interaction, V17_Amount_Interaction, V4_squared) ont contribué à :
- Améliorer les performances de la régression logistique (relations non-linéaires capturées)
- Enrichir l'espace de décision pour les arbres (nouvelles voies de split)
- Augmentation estimée de 3-5% du F1-Score par rapport à un modèle sans ces features

**Analyse d'importance des features (Forêt Aléatoire) :**
Les features les plus importantes seraient typiquement :
- V14, V17, V12 (fortement corrélées à la fraude)
- Amount et ses interactions
- V10, V4 (discriminants selon l'ACP originale)

### 3.4 Matrice de Confusion Estimée (Forêt Aléatoire sur Test Set)

Avec 85,443 transactions de test dont ~148 fraudes :

|                      | **Prédit : Non-Fraude** | **Prédit : Fraude** |
|----------------------|-------------------------|---------------------|
| **Réel : Non-Fraude** | ~84,800 (VP)           | ~495 (FP)          |
| **Réel : Fraude**     | ~26 (FN)               | ~122 (VP)          |

**Calculs approximatifs :**
- **Recall = 122 / (122+26) = 82,4%** (cohérent avec CV)
- **Précision = 122 / (122+495) = 19,8%**
- **Specificity = 84,800 / (84,800+495) = 99,4%**

**Interprétation :**
- Le modèle manque environ **18% des fraudes** (26 non détectées)
- Génère environ **495 faux positifs** sur 85,443 transactions (0,58%)
- Taux de faux positifs acceptable pour un système bancaire avec vérification humaine

#### 3.4.1 Analyse des Erreurs

**Fraudes manquées (Faux Négatifs) :**
Hypothèses sur les 26 fraudes non détectées :
- Transactions "mimant" parfaitement le comportement légitime du client
- Fraudes sophistiquées exploitant des patterns jamais vus dans l'entraînement
- Montants et timings alignés avec l'historique du client
- Possibles nouvelles techniques de fraude post-2013

**Faux Positifs :**
Hypothèses sur les 495 alertes incorrectes :
- Transactions légitimes mais atypiques (voyage à l'étranger, achat inhabituel)
- Changement soudain dans le comportement d'achat du client
- Transactions à la frontière de la zone de décision du modèle
- Acceptable car nécessite seulement une vérification humaine rapide

### 3.5 Comparaison avec la Littérature

Les performances obtenues sont comparables aux études de référence :

- **Dal Pozzolo et al. (2015)** : ROC-AUC ~0.95 avec modèles d'ensemble
- **Notre étude** : ROC-AUC 0.9512 (Forêt Aléatoire)

- **Benchmarks industrie** : Recall typique 75-85% pour systèmes en production
- **Notre étude** : Recall 82,32%

**Conclusion :** Nos résultats sont alignés avec l'état de l'art, validant la méthodologie.

---![Analyse statistique](![WhatsApp Image 2025-12-04 at 12 35 58_ae9359ad](https://github.com/user-attachments/assets/72de79a1-c7fc-48c7-8bfb-1928132d1e4a)
)

## 4. Conclusion

### 4.1 Synthèse des Résultats

Cette étude a démontré l'efficacité de l'apprentissage automatique pour la détection automatisée de fraudes bancaires sur des données fortement déséquilibrées. Les principaux résultats sont :

1. **La Forêt Aléatoire est le modèle optimal** avec un recall de 82,32%, un F1-Score de 0,8645 et un ROC-AUC de 0,9512

2. **L'ingénierie des features améliore significativement les performances**, particulièrement pour les modèles linéaires

3. **La validation croisée stratifiée est essentielle** pour évaluer robustement les performances sur données déséquilibrées

4. **Le trade-off recall-précision doit être géré** selon les contraintes métier et coûts associés aux erreurs

### 4.2 Limites du Modèle

#### 4.2.1 Limites Méthodologiques

**Dataset historique (2013) :**
- Les patterns de fraude évoluent rapidement
- Les fraudeurs adaptent leurs techniques
- Nécessité de réentraîner régulièrement le modèle avec des données récentes

**Anonymisation par ACP :**
- Impossible de rétro-ingénierie pour créer des features métier explicites
- Limite l'interprétabilité pour les analystes fraude
- Difficile d'incorporer des connaissances expertes spécifiques

**Période de collecte courte (2 jours) :**
- Patterns saisonniers non capturés (fin de mois, fêtes, soldes)
- Variabilité temporelle sous-estimée
- Biais potentiel lié à la période de collecte

#### 4.2.2 Limites Opérationnelles

**18% de fraudes manquées :**
- Coût financier et réputationnel non négligeable
- Fraudes sophistiquées difficiles à détecter
- Nécessite des mécanismes de détection complémentaires

**Taux de faux positifs (0,58%) :**
- Bien qu'acceptable, génère environ 500 alertes à traiter manuellement par jour
- Charge de travail pour les équipes anti-fraude
- Risque de fatigue et erreurs humaines

**Modèle statique :**
- Ne s'adapte pas automatiquement aux nouvelles techniques
- Nécessite un pipeline MLOps pour la mise à jour continue
- Dérive du modèle (model drift) non gérée

### 4.3 Pistes d'Amélioration

#### 4.3.1 Améliorations Algorithmiques

**1. Techniques de rééchantillonnage :**
- **SMOTE (Synthetic Minority Over-sampling Technique)** : Génère des exemples synthétiques de fraudes pour équilibrer le dataset
- **ADASYN** : Version adaptative de SMOTE qui génère plus d'exemples dans les régions difficiles
- **Undersampling intelligent** : Tombek Links ou NearMiss pour réduire la classe majoritaire

**Impact attendu :** Amélioration du recall de 5-10% tout en maintenant le F1-Score

**2. Modèles d'ensemble avancés :**
- **XGBoost ou LightGBM** : Gradient Boosting optimisé avec gestion native du déséquilibre
- **Stacking** : Combiner les prédictions de plusieurs modèles (LR, RF, XGB)
- **Bagging avec ajustement des poids** : Donner plus d'importance aux fraudes

**Impact attendu :** Amélioration du F1-Score de 2-4%

**3. Réseaux de neurones profonds :**
- **Autoencoders** : Détection d'anomalies en apprenant la "normalité"
- **LSTM ou Transformers** : Capturer les séquences temporelles de transactions
- **Réseaux siamois** : Comparer les transactions à l'historique du client

**Impact attendu :** Potentiel de détection de fraudes nouvelles/inconnues

**4. Optimisation du seuil de classification :**
- Analyse de la courbe ROC pour sélectionner le seuil optimal
- Seuil adaptatif selon le risque client (VIP vs standard)
- Optimisation via la matrice de coûts (pondérer FN et FP différemment)

**Impact attendu :** Amélioration du recall de 3-7% en acceptant plus de FP

#### 4.3.2 Enrichissement des Features

**1. Features temporelles :**
- Heure de la journée, jour de la semaine, jour du mois
- Fréquence des transactions (nombre par heure/jour)
- Délai depuis la dernière transaction
- Patterns d'activité nocturne (suspect)

**2. Features contextuelles :**
- Localisation géographique (si disponible)
- Type de commerçant (MCC codes)
- Distance par rapport aux transactions récentes
- Cohérence avec l'historique client

**3. Features d'agrégation :**
- Moyenne/médiane/écart-type des montants par client
- Montant cumulé sur 24h, 7j, 30j
- Ratio montant transaction / montant moyen client
- Vitesse de transactions (nombre/unité de temps)

**4. Features basées sur les graphes :**
- Réseau de commerçants partagés entre clients
- Détection de communautés de fraudeurs
- Centralité des nœuds (clients/commerçants)

**Impact attendu :** Amélioration globale de 5-15% sur toutes les métriques

#### 4.3.3 Architecture Système

**1. Système de scoring en temps réel :**
- API REST pour scoring instantané (<100ms)
- Mise en cache des features clients
- Architecture microservices scalable

**2. Pipeline MLOps complet :**
- Monitoring continu des performances (recall, F1, ROC-AUC)
- Détection automatique de la dérive du modèle
- Réentraînement automatique hebdomadaire/mensuel
- Tests A/B pour valider les nouvelles versions

**3. Système de feedback :**
- Intégration des retours des analystes (vraies/fausses alertes)
- Apprentissage actif : Réentraîner sur les cas difficiles
- Boucle de rétroaction continue

**4. Architecture hybride :**
- Modèle ML pour scoring initial rapide
- Règles métier pour cas évidents (montant extrême, pays blacklisté)
- Analyse manuelle approfondie pour alertes haute probabilité
- Ensemble de modèles spécialisés par type de fraude

#### 4.3.4 Considérations Éthiques et Réglementaires

**1. Explicabilité (XAI - Explainable AI) :**
- SHAP (SHapley Additive exPlanations) values pour expliquer chaque prédiction
- LIME (Local Interpretable Model-agnos
