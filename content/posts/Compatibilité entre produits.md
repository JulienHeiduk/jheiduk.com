---
title: "Compatibilité entre produits"
date: 2024-05-23
draft: false
---

## Contexte

**Papier de recherche**: [https://arxiv.org/pdf/2206.13749](https://arxiv.org/pdf/2206.13749)

**Année**: 2022

**Modèles**: LLM, Arbre de décision

**Sujet**: Compatibilité entre produits basé sur des données structurées et non structurées

**Mots-clés**: #LLM #DecsionTree #RecommenderSystem

## Introduction

L'article "Adaptive Multi-view Rule Discovery for Weakly-Supervised Compatible Products Prediction" par Rongzhi Zhang, Rebecca West, Xiquan Cui, et Chao Zhang, présente une méthode innovante pour prédire la compatibilité des produits sur les plateformes de e-commerce. Ce document, publié lors de la 28ème conférence ACM SIGKDD, aborde les défis de la prédiction de compatibilité des produits en raison des données hétérogènes et de l'absence de données d'entraînement étiquetées manuellement.

## Contexte et Problématique

### Importance de la prédiction de compatibilité

La prédiction de la compatibilité des produits est cruciale pour améliorer l'expérience utilisateur sur les plateformes de e-commerce. Une recommandation précise de produits compatibles permet aux consommateurs de faire des achats informés et de confiance, ce qui augmente la satisfaction client et potentiellement les ventes.

### Défis

1. **Données hétérogènes** : Les produits peuvent avoir des descriptions structurées (attributs de produits) et non structurées (descriptions textuelles).
2. **Absence de données étiquetées manuellement** : Le processus d'étiquetage manuel des données est coûteux et chronophage. Les données disponibles sont souvent bruyantes et incomplètes.

## Méthodologie

L'article propose AMRule, un cadre de découverte de règles multi-vues, qui combine des données structurées et non structurées pour améliorer la prédiction de compatibilité des produits.

### Principes de base

1. **Apprentissage faiblement supervisé** : Utilisation des données comportementales des utilisateurs (co-achat) comme source de supervision pour générer des instances étiquetées faiblement.
2. **Découverte de règles adaptatives** : Utilisation d'une stratégie de boosting pour découvrir de nouvelles règles en se concentrant sur les instances difficiles où le modèle actuel fait des erreurs.
3. **Intégration multi-vues** : Découverte de règles à partir des attributs structurés et des descriptions non structurées des produits.

### Étapes de la Méthodologie

1. **Découverte de règles à partir des attributs structurés** :
   - Utilisation d'arbres de décision pour générer des règles basées sur les attributs produits.
   - Les règles sont créées en combinant des attributs catégoriels et numériques pour capturer des relations complexes entre les produits.

2. **Découverte de règles à partir des descriptions non structurées** :
   - Utilisation de modèles de langage pré-entraînés (LLM) pour générer des règles basées sur des données textuelles.
   - Les descriptions des produits sont utilisées pour compléter les informations manquantes des attributs structurés.

3. **Annotation et correspondance des règles** :
   - Les règles proposées sont évaluées par des annotateurs humains pour assurer leur qualité.
   - Les règles validées sont utilisées pour étiqueter de nouvelles instances, augmentant ainsi le jeu de données d'entraînement.

4. **Apprentissage et ensemble de modèles** :
   - Utilisation d'un modèle d'ensemble pondéré où chaque modèle faible contribue au modèle final basé sur son taux d'erreur.
   - Cette approche permet d'améliorer les performances globales en intégrant des modèles complémentaires.

## Résultats Expérimentaux

Les expériences ont été menées sur des ensembles de données réelles de The Home Depot couvrant quatre catégories de produits : Éclairage, Électroménager, Salle de Bain et Outils.

### Performances

- AMRule a surpassé les méthodes de référence de 5,98 % en moyenne.
- La stratégie de boosting et la découverte de règles multi-vues ont montré une amélioration significative par rapport aux méthodes statiques et aux approches itératives.

### Analyse des composants

1. **Règles multi-vues** : L'intégration des règles basées sur les attributs structurés et les descriptions non structurées a offert des gains de performance substantiels.
2. **Découverte adaptative de règles** : La stratégie de boosting a permis de cibler les instances difficiles et d'améliorer la qualité des règles proposées.

### Étude de cas

Un exemple concret sur les outils électriques et les batteries a illustré comment AMRule découvre et propose des règles basées sur les attributs produits et les descriptions, démontrant ainsi l'efficacité de la méthode.

## Conclusion

AMRule propose une approche innovante pour la prédiction de compatibilité des produits en utilisant des règles multi-vues et une stratégie de boosting adaptative. Cette méthode améliore non seulement les performances des modèles faiblement supervisés mais offre également une meilleure interprétabilité des prédictions grâce à la découverte de règles.
