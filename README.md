# jheiduk.com

Personal data science blog by Julien Heiduk, built with [Hugo](https://gohugo.io/) and the **paper-new** theme. Deployed via Cloudflare Pages.

## Notebooks

Companion notebooks for blog articles. Each notebook is self-contained and can be run top-to-bottom.

| Notebook | Description |
|----------|-------------|
| [`gcn-pytorch-recommendation.ipynb`](notebooks/gcn-pytorch-recommendation.ipynb) | Build a Graph Convolutional Network with PyTorch Geometric to generate co-purchase item recommendations via link prediction. |
| [`vllm_qlora_finetuning.ipynb`](notebooks/vllm_qlora_finetuning.ipynb) | Fine-tune TinyLlama-1.1B with QLoRA (4-bit NF4 quantization + LoRA adapters) on Python instructions, then merge and serve the model with vLLM. Requires a GPU. |
| [`fastmcp-huggingface-hub.ipynb`](notebooks/fastmcp-huggingface-hub.ipynb) | Create a FastMCP server that exposes Hugging Face Hub models and datasets as queryable MCP resources, and test it with a FastMCP client. |
| [`tiny-llm-mcp-hub-client.ipynb`](notebooks/tiny-llm-mcp-hub-client.ipynb) | Load Qwen2.5-0.5B-Instruct locally, fetch live data from the FastMCP Hugging Face Hub server, and answer questions about the catalogue via context injection. |
| [`ragas-evaluation-tutorial.ipynb`](notebooks/ragas-evaluation-tutorial.ipynb) | Evaluate RAG pipelines with the RAGAS framework using a local Qwen2.5-7B model served by Ollama. Covers faithfulness, context precision/recall, factual correctness, semantic similarity, and custom metrics. |

## Scripts

Helper scripts used by the MCP notebooks above.

| Script | Description |
|--------|-------------|
| [`hf_server.py`](notebooks/hf_server.py) | FastMCP server exposing four MCP resources (`hf://models`, `hf://datasets`, and URI-template variants for individual model/dataset lookup) backed by the Hugging Face Hub API. |
| [`hf_client.py`](notebooks/hf_client.py) | Standalone test client that launches `hf_server.py` over stdio and reads all four resources. |

## Development

```bash
# Start dev server
hugo server

# Build for production
hugo
```