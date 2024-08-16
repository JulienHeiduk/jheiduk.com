---
title: "Manage Product Compatibility with Machine Learning"
date: 2024-05-23
draft: false
---

## Context

**Research Paper**: [https://arxiv.org/pdf/2206.13749](https://arxiv.org/pdf/2206.13749)

**Year**: 2022

**Models**: LLM, Decision Tree

**Topic**: Product compatibility based on structured and unstructured data

**Keywords**: #LLM #DecisionTree #RecommenderSystem

## Introduction

The paper "Adaptive Multi-view Rule Discovery for Weakly-Supervised Compatible Products Prediction" by Rongzhi Zhang, Rebecca West, Xiquan Cui, and Chao Zhang presents an innovative method for predicting product compatibility on e-commerce platforms. This document, published at the 28th ACM SIGKDD Conference, addresses the challenges of product compatibility prediction due to heterogeneous data and the lack of manually labeled training data.

## Background and Problem Statement

### Importance of Compatibility Prediction

Predicting product compatibility is crucial for improving user experience on e-commerce platforms. Accurate recommendations of compatible products enable consumers to make informed and confident purchases, which increases customer satisfaction and potentially boosts sales.

### Challenges

1. **Heterogeneous Data**: Products may have both structured (product attributes) and unstructured (textual descriptions) data.
2. **Lack of Manually Labeled Data**: The process of manually labeling data is expensive and time-consuming. The available data is often noisy and incomplete.

## Methodology

The paper proposes AMRule, a multi-view rule discovery framework that combines structured and unstructured data to improve product compatibility prediction.

### Core Principles

1. **Weakly Supervised Learning**: Utilizes user behavioral data (co-purchase) as a source of supervision to generate weakly labeled instances.
2. **Adaptive Rule Discovery**: Uses a boosting strategy to discover new rules by focusing on hard instances where the current model makes errors.
3. **Multi-view Integration**: Discovers rules from both structured attributes and unstructured product descriptions.

### Methodology Steps

1. **Rule Discovery from Structured Attributes**:
   - Decision trees are used to generate rules based on product attributes.
   - Rules are created by combining categorical and numerical attributes to capture complex relationships between products.

2. **Rule Discovery from Unstructured Descriptions**:
   - Utilizes pre-trained language models (LLM) to generate rules based on textual data.
   - Product descriptions are used to complement the missing information from structured attributes.

3. **Rule Annotation and Matching**:
   - Proposed rules are evaluated by human annotators to ensure their quality.
   - Validated rules are used to label new instances, thus expanding the training dataset.

4. **Learning and Ensemble Modeling**:
   - A weighted ensemble model is used where each weak model contributes to the final model based on its error rate.
   - This approach improves overall performance by integrating complementary models.

## Experimental Results

The experiments were conducted on real-world datasets from The Home Depot, covering four product categories: Lighting, Appliances, Bath, and Tools.

### Performance

- AMRule outperformed baseline methods by an average of 5.98%.
- The boosting strategy and multi-view rule discovery showed significant improvement over static methods and iterative approaches.

### Component Analysis

1. **Multi-view Rules**: Integrating rules based on structured attributes and unstructured descriptions provided substantial performance gains.
2. **Adaptive Rule Discovery**: The boosting strategy helped target hard instances and improved the quality of proposed rules.

### Case Study

A concrete example on power tools and batteries illustrated how AMRule discovers and proposes rules based on product attributes and descriptions, demonstrating the effectiveness of the method.

## Conclusion

AMRule proposes an innovative approach to product compatibility prediction by using multi-view rules and an adaptive boosting strategy. This method not only improves the performance of weakly supervised models but also offers better interpretability of predictions through rule discovery.
