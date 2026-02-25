from fastmcp import FastMCP
from huggingface_hub import HfApi
import json

mcp = FastMCP("HuggingFace Hub Explorer")
api = HfApi()


@mcp.resource("hf://models")
def list_models() -> str:
    """Top 10 most liked text-generation models on Hugging Face Hub."""
    models = list(api.list_models(filter="text-generation", sort="likes", direction=-1, limit=10))
    return json.dumps([
        {"id": m.id, "likes": m.likes or 0, "downloads": m.downloads or 0}
        for m in models
    ], indent=2)


@mcp.resource("hf://datasets")
def list_datasets() -> str:
    """Top 10 most liked datasets on Hugging Face Hub."""
    datasets = list(api.list_datasets(sort="likes", direction=-1, limit=10))
    return json.dumps([
        {"id": d.id, "likes": d.likes or 0, "downloads": d.downloads or 0}
        for d in datasets
    ], indent=2)


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
        "tags": info.tags[:15],
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


if __name__ == "__main__":
    mcp.run(transport="stdio")