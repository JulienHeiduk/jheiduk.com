---
title: "Complementary Candidate Generator with Word2Vec"
date: 2024-01-15
draft: false
---

## Context

**Research Paper**: [https://arxiv.org/pdf/2211.14982.pdf](https://arxiv.org/pdf/2211.14982.pdf)

**Year**: 2022

**Models**: Prod2Vec

**Topic**: Recommendation Engine - Candidate Generators

**Keywords**: #Complementary #DualEmbeddings #FastText #Faiss #Prod2Vec

## Introduction

Candidate generation is the first step in a "Multi-Stage" recommendation system. In this article, we will present a method for generating complementary candidates for a product. While there are numerous research papers or articles focused on generating similar candidates, there are far fewer that address a complementary approach.

Before diving into the specifics, let’s first define what a complementary candidate is. We define complementarity as follows: a candidate is complementary to a product if, in practice, it is often purchased together with this main product due to its complementary nature. The "complementary candidate" is a product that, by its function or use, naturally combines with the main product. This represents an intrinsic relationship between the two products.

## 1. The Model

### a. Prod2Vec: Vector Representation of Products

The candidate generator is based on the Word2Vec architecture, which has a dual representation (dual embeddings) for each processed element. Instead of inputting a set of words, sentences, or texts, we input pairs of products purchased together.

From customer purchase sessions, we construct the data by taking purchase pairs from all possible combinations. In other words, for a given session \(s = \{p_1, p_2, p_3,\ldots, p_n\}\), we take the Cartesian product of \(s\) with itself \(s \times s = \{(p_i, p_j) | p_i \in s, p_j \in s, p_i \neq p_j\}\). By doing this for each session, we obtain a new dataset of co-purchased pairs \(D\).

### b. Dual Embeddings for Complementarity

The neural network architecture of the Word2Vec model consists of two distinct weight matrices: the input (word) matrix and the output (context) matrix. Once training is complete and the model is optimized, it is common practice to discard the output matrix and use the input matrix as the final embeddings for the products. Product similarity search tasks are then performed using the vector representation of each product in this embedding space. While this approach works efficiently in practice, little is known about the additional information that could be gleaned from using both matrices together.

One of the key details often overlooked in Skip-Gram-Negative-Sampling (SGNS) models is that each element has two representations. For each element \(p\), the model generates two vectors \(v_p\) and \(v_p'\) corresponding to this element, where \(v_p\) and \(v_p'\) are the vectors contained in the input embedding matrix \(W_{\text{in}}\) and the output embedding matrix \(W_{\text{out}}\), respectively. Once the model is trained, users tend to discard one of the vector representations and use the other for inference, with a popular choice being to retain the vectors in the input embedding matrix \(W_{\text{in}}\).

Several studies in natural language processing have discovered that the dot product between two word embeddings from the two different matrices serves as an indicator of relevance, while the dot product from the same matrix provides a measure of similarity.

We can transpose the concept of relevance between words to the concept of complementarity between products. In particular, this relationship can be optimally illustrated using co-purchase pairs. Consider two pairs of items purchased together, \(r_1\) and \(r_2\), where \(r_1 = (\text{queen mattress}, \text{sheet})\) and \(r_2 = (\text{twin mattress}, \text{sheet})\). The resulting input matrix from the model will be able to capture that the queen mattress is similar to the twin mattress since they both appear in a similar context relative to the sheet. Similarly, the resulting matrices will also recognize that the sheet is related or complementary to both the queen and twin mattresses since they co-occur.

For the remainder of the article, we introduce the following notation to denote the different sets of recommendations generated using the input and output vectors of a given product \(p\):
- **IN-OUT**: \(\underset{v' \in W_{\text{out}}}{\text{arg max}} \cos(v_p, v')\)
- **IN-IN**: \(\underset{v \in W_{\text{in}}}{\text{arg max}} \cos(v_p, v)\)
- **OUT-OUT**: \(\underset{v' \in W_{\text{out}}}{\text{arg max}} \cos(v'_p, v')\)
- **OUT-IN**: \(\underset{v \in W_{\text{in}}}{\text{arg max}} \cos(v'_p, v)\)

To capture complementary relationships, we will rely on inference using IN-OUT. In our experiments, we do not consider the OUT-IN variant because we are interested in the direct relationship between product pairs (e.g., the sheet as a complementary item to the queen mattress, but not vice versa).

Here is an example of recommendations based on the two matrices:

![Article_3_example](/Article_3_example.png)

### Data Augmentation
One of the main challenges in using purchase data for training is its scarcity. To address this issue, we introduce synthetic product pairs based on similarity measures derived from other available data sources (clicks, product metadata, etc.).

### Inference Augmentation
During the inference phase, we apply an augmentation technique for products that are not represented in our training set. For a target product not represented \(\p'\), we identify a similar product \(p\) in the training set and use \(p\)'s recommendations for \(\p'\).

### Experiments and Results
To evaluate the effectiveness of our method, we conducted experiments on datasets from two major e-commerce companies: Overstock.com and Instacart.com.

1. Datasets
   - Overstock.com: Proprietary dataset covering two years of transactions, with over 18 million users and 24 million sessions.
   - Instacart.com: Public dataset comprising more than 3 million orders, collected across various product categories.

2. Implementation Details
   - fastText++ was used to train the SGNS model and Faiss for real-time nearest neighbor searches. Hyperparameters were optimized using Ray Tune.

3. Data Preprocessing
   - We filtered out users and sessions with an unusually high number of purchases to avoid biases. Product pairs were filtered using Pointwise Mutual Information (PMI) to identify weak relationships.

4. Evaluation of Dual Embeddings
   - The results showed that the IN-OUT approach outperformed other methods in terms of precision and recall on the tested datasets. This approach proved more effective in capturing complementary relationships between products.

## Conclusion
By using dual embeddings for complementary product recommendations, our method significantly improves the relevance of recommendations. The simplicity of implementation and the efficiency of our model make it a viable solution for large e-commerce platforms. Our experiments demonstrate an increase in product coverage and an improvement in recommendation relevance, providing a solid foundation for future research in this area.
