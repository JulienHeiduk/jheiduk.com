---
title: "Cleora part 2: How to create user embeddings?"
date: 2025-04-17
draft: false
tags: ["graph", "embeddings", "recommendation"]
summary: "This tutorial explores how to generate user embeddings from interaction data using Cleora, a high-performance graph embedding tool — ideal for recommendation systems and graph-based machine learning."
---

# Building User Embeddings with MovieLens 100k and Cleora

In this tutorial, we will demonstrate how to generate user embeddings using the MovieLens 100k dataset and Cleora, a fast and scalable embedding generation tool. The process involves collecting the dataset, preparing it for training, generating movie embeddings, and then aggregating these embeddings to create user embeddings.

## Table of Contents

1. [Collect the Data – MovieLens 100k](#1-collect-the-data--movielens-100k-get_data-collectpy)
2. [Prepare the Data for Training – Clique Expansion](#2-prepare-the-data-for-training--clique-expansion-preprocessing-preprocessingpy)
3. [Train Cleora to Generate Movie Embeddings](#3-train-cleora-to-generate-movie-embeddings-embeddings-trainingpy)
4. [Create User Embeddings from Movie Embeddings](#4-create-user-embeddings-from-movie-embeddings-embeddings-user_embeddingspy)

---

## 1. Collect the Data – MovieLens 100k (`get_data/collect.py`)

First, we will download the MovieLens 100k dataset and Cleora. The dataset contains user ratings for various movies, making it suitable for building recommender systems.

```python
import os
import subprocess

# Define the path to save the dataset
PATH = "../get_data/"
os.makedirs(PATH, exist_ok=True)

# Download the MovieLens 100k dataset
subprocess.run(["wget", "https://files.grouplens.org/datasets/movielens/ml-100k.zip", "-P", PATH], check=True)
subprocess.run(["unzip", "-d", PATH, "-o", "-q", os.path.join(PATH, "ml-100k.zip")], check=True)

# Download Cleora
subprocess.run(["wget", "https://github.com/Synerise/cleora/releases/download/v1.1.1/cleora-v1.1.1-x86_64-unknown-linux-gnu", "-P", PATH], check=True)
```

### Explanation
1. **Directory Creation**: We create a directory to store the dataset and Cleora binaries. This ensures that all necessary files are organized and accessible for later stages.
2. **Dataset Download**: The MovieLens dataset is downloaded and unzipped. This dataset is widely used for recommendation systems and provides a rich set of user-movie interactions.
3. **Cleora Download**: Cleora is downloaded for use in embedding generation. Cleora is known for its speed and scalability, which is crucial when handling large-scale graph data.

### Alternatives and Suggestions
- **Other Graph Embedding Libraries**: Consider alternatives like Node2Vec, which focuses on node embeddings in graphs, or PyTorch Geometric, which offers a more extensive suite for deep learning on graphs.
- **Use Cases**: Cleora is particularly strong in scenarios where speed and scalability are critical, such as real-time recommendation systems or large-scale graph analysis. It might be less suited for tasks that require deep learning-based embeddings or highly flexible architectures.

---

## 2. Prepare the Data for Training – Clique Expansion (`preprocessing/preprocessing.py`)

Next, we will preprocess the dataset to create a graph representation that Cleora can use for generating embeddings.

```python
import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

# Configuration parameters
config = {
    'cleora_n_iter': 5,
    'cleora_dim': 1024,
    'train_test_split': 0.2,
}

PATH = "../get_data/"
ml_path = os.path.join(PATH, "ml-100k", "u.data")

# Load the dataset
df = pd.read_csv(ml_path, sep="\t", names=["user", "movie", "rating", "timestamp"])

# Prepare user-movie interactions
with open(os.path.join(PATH, "user_movie.txt"), "w") as f:
    for row in df.itertuples():
        f.write(f"user_{row.user} movie_{row.movie}\n")

# File paths for Cleora input
movie_cleora_input_clique_filename = os.path.join(PATH, "ml-100k", "ml_clique.txt")
movie_lp_train_filename = os.path.join(PATH, "ml-100k", "ml_train.txt")
movie_lp_test_filename = os.path.join(PATH, "ml-100k", "ml_test.txt")

# Create training and testing datasets
df['user_id'] = df['user'].apply(lambda x: f"user_{x}")
df['movie_id'] = df['movie'].apply(lambda x: f"movie_{x}")

train, test = train_test_split(df[["user_id", "movie_id"]], test_size=config['train_test_split'], random_state=42)
train.columns = ["id_1", "id_2"]
test.columns = ["id_1", "id_2"]

grouped_train = train.groupby("id_1")

with open(movie_cleora_input_clique_filename, "w") as f_cleora_clique, \
     open(movie_lp_train_filename, "w") as f_train:

    for _, group in grouped_train:
        movie_ids = group["id_2"].tolist()
        f_cleora_clique.write(f"{' '.join(movie_ids)}\n")
        for i in range(len(movie_ids)):
            for j in range(i + 1, len(movie_ids)):
                f_train.write(f"{movie_ids[i]}\t{movie_ids[j]}\n")

# Write test file
with open(movie_lp_test_filename, "w") as f_test:
    grouped_test = test.groupby("id_1")
    for user, group in grouped_test:
        movie_ids = group["id_2"].tolist()
        for movie_id in movie_ids:
            f_test.write(f"{user}\t{movie_id}\n")
```

### Explanation
1. **Data Loading**: Load the MovieLens dataset into a DataFrame. This structured format allows for efficient data manipulation and analysis.
2. **User-Movie Interaction**: Create a user-movie interaction file to represent each user's interactions with movies, which is essential for graph-based embedding generation.
3. **Clique Expansion**: Format the data as cliques for Cleora input, creating both training and testing files from the user-movie interactions. This process involves creating a fully connected subgraph of movies for each user, which Cleora will use to learn embeddings. You can learn more about clique expansion [here](https://jheiduk.com/posts/cleora_graph_embeddings/).

### Alternatives and Suggestions
- **Alternative Aggregation Strategies**: Consider different strategies for aggregating movie embeddings into user embeddings, such as using a weighted mean based on ratings or applying more complex neural network-based aggregators.
- **Graph Representation**: Explore other graph representations that might capture different aspects of the data, such as bipartite graphs or hypergraphs.

---

## 3. Train Cleora to Generate Movie Embeddings (`embeddings/training.py`)

Now we will utilize Cleora to compute embeddings for movies based on the graph representation we created.

```python
import subprocess
import os

# Configuration parameters
config = {
    'cleora_n_iter': 5,
    'cleora_dim': 1024,
}

# Function to generate output filename based on columns
def columns2output_filename(output_dir, columns):
    columns_split = columns.split()
    return os.path.join(output_dir, f'emb__{"__".join(columns_split)}.out')

def train_cleora(dim, n_iter, columns, input_filename, output_dir):
    command = [os.path.join("../get_data/", 'cleora-v1.1.1-x86_64-unknown-linux-gnu'),
                '--columns', columns,
                '--dimension', str(dim),
                '-n', str(n_iter),
                '--input', input_filename,
                '-o', output_dir]
    subprocess.run(command, check=True)
    return columns2output_filename(output_dir, columns)

# Run Cleora with the specified parameters
subprocess.run(['chmod', '+x', os.path.join("../get_data/", 'cleora-v1.1.1-x86_64-unknown-linux-gnu')], check=True)

cleora_output_clique_filename = train_cleora(config['cleora_dim'], config['cleora_n_iter'], 
                                             "complex::reflexive::movie_id", 
                                             "../get_data/ml-100k/ml_clique.txt", 
                                             "../output")

# Display a sample of the embeddings
with open("../output/emb__movie_id__movie_id.out") as f:
    for _ in range(2):
        print(f.readline())
```

### Explanation
1. **Embedding Generation**: Cleora is executed to generate embeddings using the clique input file. The process involves Cleora iterating through the graph to learn vector representations for each node (movie).
2. **Output Handling**: The output is saved in a specified directory, and we print a few lines of the resulting embeddings for inspection. This step ensures that the embeddings have been correctly generated and can be used for downstream tasks.

### Alternatives and Suggestions
- **Embedding Parameters**: Experiment with different dimensions and iteration counts to see how they affect the quality of the embeddings.
- **Interpretation**: Consider visualizing embeddings using techniques like t-SNE or PCA to understand their structure and relationships.

---

## 4. Create User Embeddings from Movie Embeddings (`embeddings/user_embeddings.py`)

Finally, we will create user embeddings by aggregating the movie embeddings associated with each user.

```python
import pandas as pd

# Load movie embeddings
filepath = "../output/emb__movie_id__movie_id.out"

with open(filepath, "r") as f:
    lines = f.readlines()[1:]  # Skip the first line

# Parse embeddings
data = []
index = []
for line in lines:
    parts = line.strip().split()
    movie_id = parts[0]
    embedding = list(map(float, parts[2:]))  # Ignore the integer in second position
    index.append(movie_id)
    data.append(embedding)

# Create a DataFrame for movie embeddings
movie_emb = pd.DataFrame(data, index=index)

# Load training data
train = pd.read_csv("../get_data/ml-100k/ml_train.csv")

# Ensure all movies in train are present in embeddings
assert len(train['id_2'].unique()) == movie_emb.shape[0], \
    f"Train movies: {len(train['id_2'].unique())}, embeddings: {movie_emb.shape[0]}"

# Create user embeddings by averaging movie embeddings
train_with_emb = train.join(movie_emb, on="id_2")
user_emb = train_with_emb.groupby("id_1").mean(numeric_only=True)
user_emb = user_emb.astype("float32")

print(user_emb)
assert user_emb.shape[0] == len(train['id_1'].unique())
```

### Explanation
1. **Embedding Loading**: Load the generated movie embeddings into a DataFrame. This step is crucial for mapping user interactions to their respective embeddings.
2. **Data Integrity Check**: Ensure all movies in the training dataset have corresponding embeddings. This check prevents any loss of data integrity that could affect user embedding generation.
3. **User Embedding Creation**: Aggregate movie embeddings for each user by computing the mean, which creates a vector representation of a user based on their movie interactions.

### Alternatives and Suggestions
- **Advanced Aggregation Methods**: Explore other aggregation methods like attention mechanisms or neural network-based approaches to capture more complex user preferences.
- **Use Cases**: User embeddings can be used for personalized recommendations, user similarity analysis, and clustering users based on preferences.

---

## Conclusion

In this tutorial, we walked through the entire pipeline of generating user embeddings using the MovieLens 100k dataset and Cleora. By following these steps, you can replicate the process and explore further enhancements or applications of user embeddings in your projects. In the next articles we are going to see how to user embeddings to create a recommender systems with unsupervised approach and with a supervised approach.