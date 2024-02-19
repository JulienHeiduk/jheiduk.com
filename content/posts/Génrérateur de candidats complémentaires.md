---
title: "Génrérateur de candidats complémentaires"
date: 2024-01-15
draft: true
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

Un des détails clés souvent négligé dans les modèles Skip-Gram-Negative-Sampling (SGNS) est le fait que chaque élément possède deux représentations. Pour chaque élément $\(p\)$, le modèle génère deux vecteurs $\(v_p\)$ et $\(v'_p\)$ correspondant à cet élément, où $\(v_p\)$ et $\(v'_p\)$ sont les vecteurs contenus respectivement dans la matrice d'embedding d'entrée $\(W_{\text{in}}\)$ et la matrice d'embedding de sortie $\(W_{\text{out}}\)$. Une fois le modèle entraîné, les utilsateurs ont tendance à écarter une des représentations vectorielles et à utiliser l'autre pour l'inférence, un choix populaire étant de conserver les vecteurs dans la matrice d'embedding d'entrée $\(W_{\text{in}}\)$.

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
