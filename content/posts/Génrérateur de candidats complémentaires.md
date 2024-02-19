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
