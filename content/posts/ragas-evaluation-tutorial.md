---
title: "Evaluating RAG Pipelines with RAGAS: A Comprehensive Tutorial"
date: 2026-04-02
draft: false
summary: "RAGAS provides objective, LLM-powered metrics to evaluate every component of your RAG pipeline. Learn how to measure faithfulness, context precision, context recall, and more with Qwen2.5 served locally via Ollama — fully offline, no API key required."
---

You built a RAG pipeline. It retrieves documents, generates answers, and looks promising in a demo. But how do you *measure* whether it actually works? Manual review does not scale, and traditional NLP metrics like BLEU or ROUGE miss semantic nuance. [RAGAS](https://docs.ragas.io/en/stable/) (Retrieval Augmented Generation Assessment) is an open-source framework that gives you objective, LLM-powered evaluation metrics purpose-built for RAG systems. In this tutorial we use [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) served locally with [Ollama](https://ollama.com/) as the evaluator LLM — no paid API key required. We walk through the entire evaluation workflow, from installation to interpreting results.

## 1. Why Evaluate RAG Systems?

RAG pipelines have two failure modes that are hard to catch without systematic evaluation:

- **Retrieval failures**: the vector search returns irrelevant or incomplete context.
- **Generation failures**: the LLM hallucinates, ignores the context, or produces an off-topic response.

RAGAS tackles both with dedicated metrics for each stage of the pipeline. It decomposes evaluation into four orthogonal dimensions:

| Dimension | What it measures | Metric |
|---|---|---|
| Retrieval quality | Are the right documents retrieved? | Context Precision, Context Recall |
| Faithfulness | Is the answer grounded in the context? | Faithfulness |
| Answer quality | Is the answer relevant to the question? | Response Relevancy |
| Factual accuracy | Does the answer match the ground truth? | Factual Correctness |

## 2. Installation and Setup

```bash
pip install ragas openai sentence-transformers
```

> **Why `openai` in the dependencies?** RAGAS's evaluation metrics need structured JSON output from the LLM. Under the hood this is handled by the [Instructor](https://github.com/jxnl/instructor) library, which speaks the "OpenAI-compatible" HTTP protocol — an open standard also implemented by [Ollama](https://ollama.com/). The `openai` pip package is just the HTTP client; **no data leaves your machine**.

RAGAS uses LLMs as evaluators. Most tutorials call a proprietary API, but you can serve any open-source model locally instead. In this tutorial we run [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) on your own machine with [Ollama](https://ollama.com/) and point RAGAS at `localhost`. No account, no token, fully offline. Works on Windows, macOS, and Linux.

> **Why not a 3B model?** RAGAS metrics need the evaluator LLM to produce structured JSON output (claim extraction, relevance judgments). Models under ~7B parameters struggle with this and produce parsing errors. Qwen2.5-7B hits the sweet spot: small enough to run on a consumer GPU (~4.7 GB quantised), large enough to reliably follow structured output schemas.

### Step 1 — Install Ollama and pull Qwen2.5

Download Ollama from [ollama.com](https://ollama.com/) (one-click installer for Windows/macOS/Linux). Then pull the model:

```bash
ollama pull qwen2.5:7b
```

Ollama starts automatically after installation and serves on `http://localhost:11434`. The first pull downloads the quantised weights (~4.7 GB).

### Step 2 — Connect RAGAS to Ollama

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory

client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",  # local Ollama server
    api_key="unused",                      # required by the client, ignored by Ollama
)
llm = llm_factory("qwen2.5:7b", client=client)
```

That is it — the same evaluation code works whether RAGAS talks to a cloud API or to a local Ollama server. The only thing that changes is `base_url`.

## 3. Understanding the Core Metrics

### 3.1 Faithfulness

Faithfulness measures how factually consistent a response is with the retrieved context. The formula is straightforward:

$$\text{Faithfulness} = \frac{\text{Number of claims supported by context}}{\text{Total claims in the response}}$$

A score of 1.0 means every claim in the response can be traced back to the retrieved documents. A low score signals hallucination.

```python
from ragas.metrics.collections import Faithfulness

scorer = Faithfulness(llm=llm)

result = await scorer.ascore(
    user_input="When was the first Super Bowl?",
    response="The first Super Bowl was held on January 15, 1967. It was played in Los Angeles.",
    retrieved_contexts=[
        "The First AFL-NFL World Championship Game, later known as Super Bowl I, "
        "was played on January 15, 1967, at the Los Angeles Memorial Coliseum."
    ],
)
print(f"Faithfulness: {result.value}")
# Faithfulness: 1.0 — both claims are supported by the context
```

Now inject a hallucination and watch the score drop:

```python
result = await scorer.ascore(
    user_input="When was the first Super Bowl?",
    response="The first Super Bowl was held on January 15, 1967. It was watched by 200 million viewers.",
    retrieved_contexts=[
        "The First AFL-NFL World Championship Game was played on January 15, 1967, "
        "at the Los Angeles Memorial Coliseum in front of 61,946 spectators."
    ],
)
print(f"Faithfulness: {result.value}")
# Faithfulness: 0.5 — the "200 million viewers" claim is not supported
```

### 3.2 Context Precision

Context Precision evaluates the retriever's ability to rank relevant chunks higher than irrelevant ones. It computes a weighted mean precision@k — placing an irrelevant document at position 1 is penalized much more harshly than at position 5.

```python
from ragas.metrics.collections import ContextPrecision

scorer = ContextPrecision(llm=llm)

# Good ranking: relevant context first
result_good = await scorer.ascore(
    user_input="Where is the Eiffel Tower located?",
    reference="The Eiffel Tower is located in Paris, France.",
    retrieved_contexts=[
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
        "It was named after engineer Gustave Eiffel.",
        "Paris is the capital of France.",
    ],
)
print(f"Context Precision (good ranking): {result_good.value}")
# ~0.99

# Bad ranking: irrelevant context first
result_bad = await scorer.ascore(
    user_input="Where is the Eiffel Tower located?",
    reference="The Eiffel Tower is located in Paris, France.",
    retrieved_contexts=[
        "The Statue of Liberty is in New York City.",
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
        "Paris is the capital of France.",
    ],
)
print(f"Context Precision (bad ranking): {result_bad.value}")
# ~0.50 — irrelevant chunk ranked first tanks the score
```

### 3.3 Context Recall

Context Recall measures retrieval completeness — how many claims from the reference answer are supported by the retrieved documents.

$$\text{Context Recall} = \frac{\text{Reference claims supported by retrieved context}}{\text{Total claims in reference}}$$

```python
from ragas.metrics.collections import ContextRecall

scorer = ContextRecall(llm=llm)

# Complete retrieval
result = await scorer.ascore(
    user_input="Tell me about the Eiffel Tower.",
    reference="The Eiffel Tower is in Paris. It is 330 meters tall. It was built in 1889.",
    retrieved_contexts=[
        "The Eiffel Tower is located in Paris, France.",
        "The tower stands 330 meters tall and was completed in 1889.",
    ],
)
print(f"Context Recall: {result.value}")
# ~1.0 — all three claims from the reference are covered

# Incomplete retrieval
result = await scorer.ascore(
    user_input="Tell me about the Eiffel Tower.",
    reference="The Eiffel Tower is in Paris. It is 330 meters tall. It was built in 1889.",
    retrieved_contexts=[
        "The Eiffel Tower is located in Paris, France.",
    ],
)
print(f"Context Recall: {result.value}")
# ~0.33 — only 1 out of 3 reference claims is supported
```

### 3.4 Factual Correctness

Factual Correctness compares a response directly against a reference answer by decomposing both into claims and computing precision, recall, or F1:

```python
from ragas.metrics.collections import FactualCorrectness

scorer = FactualCorrectness(llm=llm)

result = await scorer.ascore(
    response="The Eiffel Tower is in Paris. It is 300 meters tall.",
    reference="The Eiffel Tower is in Paris. It is 330 meters tall. It was built in 1889.",
)
print(f"Factual Correctness (F1): {result.value}")
# ~0.67 — correct on location, wrong on height, missing construction date
```

You can control the evaluation mode:

```python
# Precision-only: are the response claims correct?
scorer_p = FactualCorrectness(llm=llm, mode="precision")

# Recall-only: are all reference claims covered?
scorer_r = FactualCorrectness(llm=llm, mode="recall")
```

## 4. Semantic Similarity (Non-LLM Metric)

Not every metric requires an LLM call. Semantic Similarity uses embedding cosine similarity — faster and cheaper. We use `sentence-transformers` locally so there is zero API cost:

```python
from ragas.embeddings import embedding_factory
from ragas.metrics.collections import SemanticSimilarity

embeddings = embedding_factory("huggingface", model="sentence-transformers/all-MiniLM-L6-v2")
scorer = SemanticSimilarity(embeddings=embeddings)

result = await scorer.ascore(
    response="The Eiffel Tower is located in Paris.",
    reference="The Eiffel Tower is a landmark in Paris, France, standing 330 meters tall.",
)
print(f"Semantic Similarity: {result.value}")
# ~0.82
```

Because embeddings run locally on CPU, this metric adds no latency from network round-trips and works fully offline.

## 5. Custom Evaluation with Aspect Critic

RAGAS lets you define your own binary evaluation criteria with `DiscreteMetric`. This is powerful for domain-specific quality checks:

```python
from ragas.metrics import DiscreteMetric

safety_scorer = DiscreteMetric(
    name="response_safety",
    allowed_values=["safe", "unsafe"],
    prompt="""Evaluate whether the following response contains harmful, 
offensive, or dangerous content.

Response: {response}

Answer with only 'safe' or 'unsafe'.""",
)

result = await safety_scorer.ascore(
    llm=llm,
    response="To improve your sleep, try maintaining a consistent schedule and avoiding screens before bed.",
)
print(f"Safety: {result.value}")
# Safety: safe
```

You can create aspect critics for any dimension: conciseness, formality, technical accuracy, or domain compliance.

## 6. End-to-End Evaluation Pipeline

Let's bring it all together and evaluate a batch of RAG outputs systematically:

```python
import asyncio
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    Faithfulness,
    ContextPrecision,
    ContextRecall,
    FactualCorrectness,
)

client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="unused")
llm = llm_factory("qwen2.5:7b", client=client)

# Define the evaluation dataset
eval_dataset = [
    {
        "user_input": "What is transfer learning?",
        "response": (
            "Transfer learning is a technique where a model trained on one task "
            "is reused as the starting point for a model on a second task."
        ),
        "reference": (
            "Transfer learning involves taking a pre-trained model and adapting it "
            "to a new, related task. It reduces training time and data requirements."
        ),
        "retrieved_contexts": [
            "Transfer learning is a machine learning method where a model developed "
            "for one task is reused as the starting point for a model on a second task.",
            "It is popular in deep learning because it allows leveraging large "
            "pre-trained models like BERT and ResNet.",
        ],
    },
    {
        "user_input": "What is gradient descent?",
        "response": (
            "Gradient descent is an optimization algorithm that minimizes a loss function "
            "by iteratively moving in the direction of steepest descent."
        ),
        "reference": (
            "Gradient descent is an iterative optimization algorithm used to minimize "
            "a function by moving in the direction of the negative gradient."
        ),
        "retrieved_contexts": [
            "Gradient descent is a first-order optimization algorithm. It finds "
            "a local minimum by taking steps proportional to the negative gradient.",
        ],
    },
]

# Initialize scorers
scorers = {
    "faithfulness": Faithfulness(llm=llm),
    "context_precision": ContextPrecision(llm=llm),
    "context_recall": ContextRecall(llm=llm),
    "factual_correctness": FactualCorrectness(llm=llm),
}


async def evaluate_sample(sample: dict) -> dict:
    """Evaluate a single sample across all metrics."""
    results = {}
    results["faithfulness"] = await scorers["faithfulness"].ascore(
        user_input=sample["user_input"],
        response=sample["response"],
        retrieved_contexts=sample["retrieved_contexts"],
    )
    results["context_precision"] = await scorers["context_precision"].ascore(
        user_input=sample["user_input"],
        reference=sample["reference"],
        retrieved_contexts=sample["retrieved_contexts"],
    )
    results["context_recall"] = await scorers["context_recall"].ascore(
        user_input=sample["user_input"],
        reference=sample["reference"],
        retrieved_contexts=sample["retrieved_contexts"],
    )
    results["factual_correctness"] = await scorers["factual_correctness"].ascore(
        response=sample["response"],
        reference=sample["reference"],
    )
    return {k: v.value for k, v in results.items()}


async def run_evaluation():
    """Evaluate all samples and print a summary."""
    all_results = []
    for i, sample in enumerate(eval_dataset):
        scores = await evaluate_sample(sample)
        all_results.append(scores)
        print(f"\nSample {i+1}: {sample['user_input']}")
        for metric, score in scores.items():
            print(f"  {metric}: {score}")

    # Compute averages
    print("\n--- Average Scores ---")
    for metric in scorers:
        avg = sum(r[metric] for r in all_results) / len(all_results)
        print(f"  {metric}: {avg:.3f}")


asyncio.run(run_evaluation())
```

Expected output:

```
Sample 1: What is transfer learning?
  faithfulness: 1.0
  context_precision: 1.0
  context_recall: 0.67
  factual_correctness: 0.8

Sample 2: What is gradient descent?
  faithfulness: 1.0
  context_precision: 1.0
  context_recall: 1.0
  factual_correctness: 1.0

--- Average Scores ---
  faithfulness: 1.000
  context_precision: 1.000
  context_recall: 0.833
  factual_correctness: 0.900
```

## 7. Interpreting Results and Debugging

When scores are low, here is how to diagnose the problem:

| Low metric | Root cause | Fix |
|---|---|---|
| Faithfulness < 0.8 | LLM is hallucinating beyond context | Tighten the prompt ("only use the provided context"), lower temperature |
| Context Precision < 0.7 | Retriever returns noisy results | Tune chunk size, improve embeddings, add metadata filtering |
| Context Recall < 0.7 | Retriever misses relevant documents | Increase top-k, use hybrid search (BM25 + dense), re-chunk documents |
| Factual Correctness < 0.7 | Answer diverges from ground truth | Check if the reference is complete, review the LLM prompt template |

A practical workflow: run RAGAS on every commit that touches the RAG pipeline, store scores in a dashboard, and set CI thresholds (e.g., faithfulness > 0.9) to catch regressions automatically.

## 8. Key Takeaways

- **RAGAS decouples retrieval evaluation from generation evaluation** — you can pinpoint whether a failure comes from the retriever or the LLM.
- **LLM-as-judge metrics** (Faithfulness, Context Precision/Recall) capture semantic meaning that traditional metrics miss.
- **Open-source evaluators work** — Qwen2.5-7B served locally with Ollama is a fully offline, free alternative to proprietary APIs.
- **Non-LLM metrics** (Semantic Similarity, BLEU, ROUGE) are available for cost-sensitive or latency-sensitive scenarios.
- **Custom metrics** via `DiscreteMetric` and `AspectCritic` let you evaluate domain-specific quality dimensions.
- **Integrate RAGAS into CI/CD** to catch regression before deployment — treat evaluation scores like test coverage.

RAGAS transforms RAG evaluation from a subjective "looks good" into a repeatable, quantifiable process. Start with the four core metrics (Faithfulness, Context Precision, Context Recall, Factual Correctness) and expand to custom metrics as your pipeline matures.

The companion notebook with all the code from this article is available [on GitHub](https://github.com/JulienHeiduk/jheiduk.com/tree/main/notebooks/ragas-evaluation-tutorial.ipynb).
