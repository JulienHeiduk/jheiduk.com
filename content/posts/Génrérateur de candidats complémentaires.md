---
title: "Génrérateur de candidats complémentaires"
date: 2024-01-15
draft: false
---

## Contexte

**Papier de recherche**: [https://arxiv.org/pdf/2211.14982.pdf](https://arxiv.org/pdf/2211.14982.pdf)

**Année**: 2022

**Modèles**: Prod2Vec

**Sujet**: Moteur de recommandations - Générateurs de candidats

**Mots-clés**: #Complementary #DualEmbeddings #FastText #Faiss #Prod2Vec

## Introduction

La génération de candidats constitue la première étapes d'un système de recommendations "Multi-Stage". Dans cet article nous allons présenter une méthode permettant de générer des candidats complémentaires à un produit. Il existe de nombreux papier de recherche ou article permettant de générer des candidats similaires mais beaucoup moins en ce qui concerne une approche basée sur la complémentarité.

Avant de rentrer dans le vif du sujet nous allons tout d'abord définir ce qu'est un candidat complémentaire. Nous définirons la complémentarité de la manière suivante : un candidat est complémentaire à un produit si, en pratique, il est souvent acheté conjointement avec ce produit principal en raison de sa nature complémentaire. Le "candidat complémentaire" est un produit qui, de par sa fonction ou son usage, se combine naturellement avec le produit principal. Il s'agit d'une relation intrinsèque entre les deux produits.

## 1.Le modèle

### a. Prod2Vec: Représentation vectorielle des produits

Le générateur de candidat se base sur l'architecture Word2Vec qui possède une double représentation (dual embeddings) pour chaque élément traité. Au lieu de mettre dans la donnée un ensemble de mot, de phrases ou de textes nous y mettrons les pairs de produits achetés ensemble.

A partir des sessions d'achats des clients nous constiturons la donnée en prenant les paires d'achats en prenant toutes les combinaisons par paires. Autrement dit, pour une session donnée $\(s = \{p_1, p_2, p_3,\ldots, p_n\}\)$, nous prenons le produit cartésien de $\(s\)$ avec elle-même $\(s \times s = \{(p_i, p_j) | p_i \in s, p_j \in s, p_i \neq p_j\}\)$. En procédant ainsi pour chaque session, nous obtenons une nouvelle donnée de co-achat par paire $\(D\)$.

### b. Dual embeddings pour la complémentarité

L'architecture du réseau de neurones du modèle Word2Vec est composée de deux matrices de poids distinctes: la matrice d'entrée (mot) et la matrice de sortie (contexte). Une fois l'entraînement terminé et le modèle optimisé, il est d'usage courant de se défaire de la matrice de sortie et d'utiliser la matrice d'entrée comme les embeddings finaux des produits. Les tâches de recherche de similarité entre les produits sont ensuite effectuées à travers la représentation vectorielle de chaque produit dans cet espace d'embedding. Bien que cela fonctionne efficacement en pratique, on connaît peu de choses sur les informations supplémentaires l'utilisation conjoint de ces deux matrices.

Un des détails clés souvent négligé dans les modèles Skip-Gram-Negative-Sampling (SGNS) est le fait que chaque élément possède deux représentations. Pour chaque élément $\(p\)$, le modèle génère deux vecteurs $\(v_p\)$ et $\(v'_p\)$ correspondant à cet élément, où $\(v_p\)$ et $\(v`_p\)$ sont les vecteurs contenus respectivement dans la matrice d'embedding d'entrée $\(W_{\text{in}}\)$ et la matrice d'embedding de sortie $\(W_{\text{out}}\)$. Une fois le modèle entraîné, les utilsateurs ont tendance à écarter une des représentations vectorielles et à utiliser l'autre pour l'inférence, un choix populaire étant de conserver les vecteurs dans la matrice d'embedding d'entrée $\(W_{\text{in}}\)$.

Plusieurs études en traitement automatique du langage naturel ont découvert que le produit scalaire entre deux embeddings de mots issus des deux matrices différentes sert d'indicateur de pertinence, tandis que celui issu de la même matrice fournit une mesure de similarité.

Nous pouvons transposer le concept de pertinence entre les mots au concept de complémentarité entre les produits. En particulier, cette relation peut être illustrée de manière optimale en utilisant des paires de co-achats. Considérons deux paires d'articles achetés ensemble, $\(r_1\)$ et $\(r_2\)$, où $\(r_1 = (\text{matelas queen}, \text{drap})\)$ et $\(r_2 = (\text{matelas twin}, \text{drap})\)$. La matrice d'entrée résultante du modèle sera capable de capter que le matelas queen est similaire au matelas twin puisqu'ils apparaissent tous deux dans un contexte similaire par rapport au drap. De la même manière, les matrices résultantes pourront également reconnaître que le drap est lié ou complémentaire au matelas queen et au matelas twin puisqu'ils co-occurrent.

Pour le reste de l'article, nous introduisons la notation suivante pour désigner les différents ensembles de recommandations générées à l'aide des vecteurs d'entrée et de sortie d'un produit donné $\(p\)$ :
- **IN-OUT** : $\(\underset{v' \in W_{\text{out}}}{\text{arg max}} \cos(v_p, v')\)$
- **IN-IN** : $\(\underset{v \in W_{\text{in}}}{\text{arg max}} \cos(v_p, v)\)$
- **OUT-OUT** : $\(\underset{v' \in W_{\text{out}}}{\text{arg max}} \cos(v'_p, v')\)$
- **OUT-IN** : $\(\underset{v \in W_{\text{in}}}{\text{arg max}} \cos(v'_p, v)\)$

Pour saisir les relations de complémentarité, nous nous appuierons sur l'inférence en utilisant IN-OUT. Dans nos expériences, nous ne considérons pas la variante OUT-IN car nous nous intéressons à la relation directe entre les paires de produits (par exemple, le drap comme article complémentaire au matelas queen mais pas l'inverse).

Voici un exemple de recommendations en se basant sur les deux matrices: 

![Article_3_example](/Article_3_example.png)

### Augmentation des données
L'un des principaux défis dans l'utilisation des données d'achats pour l'entraînement est leur rareté. Pour pallier ce problème, nous introduisons des paires de produits synthétiques basées sur des mesures de similarité dérivées d'autres sources de données disponibles (clics, métadonnées de produits, etc.).

### Augmentation des inférences
Lors de la phase d'inférence, nous appliquons une technique d'augmentation pour les produits qui ne sont pas représentés dans notre ensemble d'entraînement. Pour un produit cible non représenté $\p'$, nous identifions un produit similaire $p$ dans l'ensemble d'entraînement et utilisons les recommandations de $p$ pour $\p'$.

### Expériences et résultats
Pour évaluer l'efficacité de notre méthode, nous avons réalisé des expériences sur des ensembles de données provenant de deux grandes entreprises de commerce en ligne : Overstock.com et Instacart.com.

1. Ensembles de données
Overstock.com : Ensemble de données propriétaire comprenant deux ans de transactions, avec plus de 18 millions d'utilisateurs et 24 millions de sessions.
Instacart.com : Ensemble de données public comprenant plus de 3 millions de commandes, collectées à partir de diverses catégories de produits.
2. Détails de la mise en œuvre
Nous avons utilisé fastText++ pour entraîner le modèle SGNS et Faiss pour les recherches de plus proches voisins en temps réel. Les hyperparamètres ont été optimisés à l'aide de Ray Tune.

3. Pré-traitement des données
Nous avons éliminé les utilisateurs et sessions avec un nombre exceptionnellement élevé d'achats pour éviter les biais. Les paires de produits ont été filtrées en utilisant la PMI (Pointwise Mutual Information) pour identifier les relations faibles.

4. Évaluation des embeddings duaux
Les résultats ont montré que l'approche IN-OUT surpassait les autres méthodes en termes de précision et de rappel sur les ensembles de données testées. Cette approche s'est avérée plus efficace pour capturer les relations de complémentarité entre les produits.

## Conclusion
En utilisant des embeddings duaux pour les recommandations de produits complémentaires, notre méthode améliore significativement la pertinence des recommandations. La simplicité de mise en œuvre et l'efficacité de notre modèle en font une solution viable pour les grandes plateformes de commerce en ligne. Nos expérimentations démontrent une augmentation de la couverture des produits et une amélioration de la pertinence des recommandations, ce qui constitue une base solide pour des recherches futures dans ce domaine.


