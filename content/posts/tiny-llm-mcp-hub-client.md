---
title: "Querying the Hugging Face Hub with a Tiny LLM and FastMCP"
date: 2026-02-25
draft: false
summary: "Load Qwen2.5-0.5B-Instruct, connect it to a FastMCP server via context injection, and answer live questions about the Hugging Face Hub catalogue."
---

Running a 500-million-parameter model as an intelligent agent sounds ambitious. Pair it with structured context from an MCP server and it becomes surprisingly capable: the model handles reasoning, the server handles data access, and neither needs to know how the other works internally. This tutorial builds on the [FastMCP Hugging Face Hub server](https://jheiduk.com/posts/fastmcp-huggingface-hub/) — four MCP resources exposing Hub model and dataset metadata — and shows how to wire **Qwen2.5-0.5B-Instruct** to it so the model can answer live questions about the catalogue without any parametric knowledge of the Hub.

## Prerequisites

```bash
pip install fastmcp huggingface_hub transformers torch nest_asyncio
```

The server from the previous article must be available as `hf_server.py`. If not, run the companion notebook to regenerate it — the server is a single file written by one cell.

## Step 1: Load the Model

**Qwen2.5-0.5B-Instruct** is instruction-tuned, ships with a chat template, and fits comfortably in CPU memory. It downloads roughly 1 GB on first use and is cached by `huggingface_hub`.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,   # use torch.bfloat16 if you have a GPU
)
model.eval()
```

`model.eval()` disables dropout so inference is deterministic. The choice of 0.5B is intentional: when context is provided explicitly, parametric knowledge matters less, and a smaller model answers faster with far less memory.

## Step 2: Read a Resource from the MCP Server

`PythonStdioTransport` spawns `hf_server.py` as a subprocess with an explicit `log_file`, which avoids the `fileno` `UnsupportedOperation` error that occurs when Jupyter's `ipykernel` `OutStream` is passed as stderr to `subprocess.Popen`. `nest_asyncio` lets `asyncio.run()` work inside Jupyter's existing event loop. `read_resource` returns a list of content objects; `.text` gives the raw string payload.

```python
import asyncio
import sys

import nest_asyncio
from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport

nest_asyncio.apply()


async def fetch_resource(uri: str) -> str:
    transport = PythonStdioTransport(
        script_path="hf_server.py",
        python_cmd=sys.executable,
        log_file=open("mcp_server.log", "w"),  # real file → has fileno()
    )
    async with Client(transport) as client:
        result = await client.read_resource(uri)
    return result[0].text


catalogue = asyncio.run(fetch_resource("hf://models"))
print(catalogue[:300])
```

The catalogue is a compact JSON array — `id`, `likes`, `downloads` per model. That structure is deliberate: small models parse clean JSON reliably; verbose free-text descriptions add tokens without adding signal.

## Step 3: Inject Context and Generate an Answer

The pattern is **context injection**: the MCP resource goes into the system message, the user's question goes in the user turn. The model never calls the MCP server itself — the orchestration layer fetches the data and hands it to the model as plain text.

```python
def ask(question: str, context: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with access to a Hugging Face Hub catalogue.\n\n"
                f"Catalogue (JSON):\n{context}"
            ),
        },
        {"role": "user", "content": question},
    ]

    # apply_chat_template formats messages with Qwen2.5's special tokens
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,              # greedy for reproducibility
            pad_token_id=tokenizer.eos_token_id,
        )

    # slice off the echoed prompt tokens before decoding
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
```

`apply_chat_template` inserts the `<|im_start|>` / `<|im_end|>` tokens that Qwen2.5 expects. Without them the model produces incoherent output. Slicing `output_ids` from `inputs["input_ids"].shape[1]` strips the prompt echo so only the generated answer is decoded.

## Step 4: Full Pipeline

The demo fetches two different resources and asks a question about each. Because `fetch_resource` is async, the natural approach is a single `async def main` that awaits both reads before calling the synchronous `ask` function.

```python
async def main():
    # 1. Top models catalogue → find the most downloaded
    catalogue = await fetch_resource("hf://models")
    answer = ask(
        "Which model in the catalogue has the most downloads? Give the model ID and the download count.",
        catalogue,
    )
    print("Most downloaded model:")
    print(answer)

    # 2. Specific model detail → extract task and tags
    detail = await fetch_resource("hf://models/google/gemma-2-2b")
    answer2 = ask(
        "What task is this model designed for? List three of its tags.",
        detail,
    )
    print("\nGemma-2-2b detail:")
    print(answer2)


asyncio.run(main())
```

<!-- Diagram: Three horizontal boxes connected by arrows. Left box: "Qwen2.5-0.5B-Instruct" (inference). Middle box: "Python orchestrator" (fetch_resource + ask). Right box: "FastMCP Server" (hf://models, hf://models/{owner}/{name}). Arrow from orchestrator to FastMCP: "read_resource()". Arrow back: "JSON string". Arrow from orchestrator to LLM: "system prompt + context". Arrow back: "generated answer". -->
![tiny-llm-mcp-architecture](/tiny-llm-mcp-architecture.svg)
*Figure: The orchestrator fetches data from the MCP server and injects it into the LLM's context; the model never calls the server directly.*

With the catalogue in the system prompt, Qwen2.5-0.5B reliably identifies the most-downloaded model and extracts structured fields from the detail JSON — both tasks require only careful reading, not world knowledge.

## Conclusion

Context injection cleanly separates data access from reasoning. The FastMCP server owns the interface to the Hugging Face Hub; the LLM owns interpretation. Replacing the server with a different data source — an internal model registry, a W&B experiment tracker, a Weights & Biases artifact store — requires no changes to the inference code. Swapping the model requires no changes to the server.

Three natural next steps:

- **Dynamic tool use**: expose `api.list_models(search=query)` as an `@mcp.tool()` so the model can issue targeted searches rather than receiving a static catalogue.
- **Larger models**: the same orchestration code works with any Hugging Face model; a 7B instruction-tuned model produces richer, more reliable answers with identical context.
- **Streaming**: pass a `TextStreamer` to `model.generate` to stream tokens as they are produced, which improves perceived latency for interactive applications.

For the server implementation, see [FastMCP Server with Hugging Face Hub Resources](https://jheiduk.com/posts/fastmcp-huggingface-hub/).
