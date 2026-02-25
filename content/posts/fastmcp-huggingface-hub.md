---
title: "FastMCP Server with Hugging Face Hub Resources"
date: 2026-02-25
draft: false
summary: "Build a FastMCP server that exposes Hugging Face Hub models and datasets as queryable MCP resources for LLM agents."
---

The [Model Context Protocol](https://jheiduk.com/posts/mcp/) standardises how LLMs access external data. **Resources** — read-only, URI-addressed endpoints — are the passive side of MCP: the agent reads them without triggering side effects. Hugging Face Hub is a natural source of resources for AI applications: it catalogs hundreds of thousands of models and datasets through a clean Python API. This tutorial builds a **FastMCP** server that wraps the Hub's metadata API into four MCP resources, queryable by any agent that speaks MCP.

## Prerequisites

Install the required packages:

```bash
pip install "fastmcp>=2.0" huggingface_hub
```

No Hugging Face token is needed for public metadata queries. For private or gated repositories, set the `HF_TOKEN` environment variable.

## Step 1: Initialise the Server

Create `hf_server/server.py` and set up the two core objects:

```python
from fastmcp import FastMCP
from huggingface_hub import HfApi
import json

mcp = FastMCP("HuggingFace Hub Explorer")
api = HfApi()
```

`FastMCP` handles the MCP wire protocol, tool registration, and resource routing. `HfApi` is the official client for the Hugging Face Hub REST API. Both are kept at module level so they are shared across all resource handlers without being re-created on every request.

## Step 2: Static Resources — Trending Lists

In FastMCP, a **static resource** has a fixed URI and returns a value on every read — analogous to a cacheable HTTP GET endpoint. The `@mcp.resource` decorator registers the function and associates it with the given URI.

```python
@mcp.resource("hf://models")
def list_models() -> str:
    """Top 10 trending text-generation models on Hugging Face Hub."""
    models = list(api.list_models(
        task="text-generation",
        sort="trending",
        limit=10,
    ))
    return json.dumps([
        {"id": m.id, "likes": m.likes or 0, "downloads": m.downloads or 0}
        for m in models
    ], indent=2)


@mcp.resource("hf://datasets")
def list_datasets() -> str:
    """Top 10 trending datasets on Hugging Face Hub."""
    datasets = list(api.list_datasets(
        sort="trending",
        limit=10,
    ))
    return json.dumps([
        {"id": d.id, "likes": d.likes or 0, "downloads": d.downloads or 0}
        for d in datasets
    ], indent=2)
```

`HfApi.list_models()` returns a lazy iterator; wrapping it in `list()` materialises the results before the comprehension runs. MCP resources must return a string, bytes, or blob — `json.dumps` is the natural serialisation for structured data that the LLM will read.

## Step 3: Resource Templates — Model and Dataset Detail

**Resource templates** parameterise a URI with named placeholders. FastMCP maps each `{variable}` in the URI to a function argument of the same name and resolves the correct handler at request time — no routing code required.

Repository identifiers on Hugging Face follow an `owner/name` pattern (`meta-llama/Llama-3.1-8B`, `google/gemma-2-2b`). Splitting the identifier into two path segments keeps the URI readable without URL-encoding the slash:

```python
@mcp.resource("hf://models/{owner}/{name}")
def get_model(owner: str, name: str) -> str:
    """Fetch metadata for a specific model from Hugging Face Hub."""
    info = api.model_info(f"{owner}/{name}")
    return json.dumps({
        "id": info.id,
        "author": info.author,
        "task": info.pipeline_tag or "unknown",
        "likes": info.likes,
        "downloads": info.downloads,
        "tags": info.tags[:15],  # cap to avoid oversized context payloads
    }, indent=2)


@mcp.resource("hf://datasets/{owner}/{name}")
def get_dataset(owner: str, name: str) -> str:
    """Fetch metadata for a specific dataset from Hugging Face Hub."""
    info = api.dataset_info(f"{owner}/{name}")
    return json.dumps({
        "id": info.id,
        "author": info.author,
        "likes": info.likes,
        "downloads": info.downloads,
        "tags": info.tags[:15],
    }, indent=2)
```

Tags are capped at 15 to avoid flooding the LLM's context window with exhaustive metadata. An agent querying `hf://models/mistralai/Mistral-7B-Instruct-v0.3` receives full model metadata with a single resource read.

<!-- Diagram: Three columns. Left: "LLM Agent" box. Middle: "FastMCP Server" box listing the four resources (hf://models, hf://datasets, hf://models/{owner}/{name}, hf://datasets/{owner}/{name}). Right: "HuggingFace Hub API" box. Arrows: agent → server (MCP read_resource), server → HF Hub (REST API call), HF Hub → server (JSON response), server → agent (MCP content). -->
![fastmcp-hf-hub-architecture](/fastmcp-hf-hub-architecture.svg)
*Figure: Request flow from an LLM agent through the FastMCP server to the Hugging Face Hub API.*

## Step 4: Run the Server

Add the entrypoint and start the server over stdio:

```python
if __name__ == "__main__":
    mcp.run(transport="stdio")
```

```bash
fastmcp run hf_server/server.py
```

## Step 5: Test with the FastMCP Client

Create `hf_server/client.py` to exercise all four resources:

```python
import asyncio
from fastmcp import Client

async def main():
    async with Client("hf_server/server.py") as client:
        # Trending model catalogue
        result = await client.read_resource("hf://models")
        print("Trending models:\n", result[0].text[:400])

        # Specific model detail
        result = await client.read_resource("hf://models/google/gemma-2-2b")
        print("\nModel detail:\n", result[0].text)

        # Trending dataset catalogue
        result = await client.read_resource("hf://datasets")
        print("\nTrending datasets:\n", result[0].text[:400])

        # Specific dataset detail
        result = await client.read_resource("hf://datasets/rajpurkar/squad")
        print("\nDataset detail:\n", result[0].text)

asyncio.run(main())
```

`Client("hf_server/server.py")` launches the server as a child process over stdio when the async context opens and shuts it down on exit. `read_resource` accepts any URI the server advertises; for text content, the first element of the returned list exposes a `.text` attribute.

## Conclusion

Four decorators are enough to give any MCP-compatible agent structured, on-demand access to the entire Hugging Face Hub catalogue. The server pattern scales to other read-heavy APIs: swap `HfApi` for a Weights & Biases client, an internal model registry, or any REST API and the structure stays identical.

Three natural extensions from here:

- **Search tools**: expose `api.list_models(search=query)` as an `@mcp.tool()` so agents can discover models by keyword rather than browsing the static catalogue.
- **Model cards**: use `api.hf_hub_download(repo_id, "README.md")` to serve the full model card Markdown as a resource.
- **Private repos**: pass `token=os.getenv("HF_TOKEN")` to `HfApi()` to access gated or private repositories without modifying the resource logic.

For the MCP fundamentals — tools, prompts, and the wire protocol — see the [introductory FastMCP article](https://jheiduk.com/posts/mcp/).
