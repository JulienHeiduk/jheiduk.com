---
title: "Model Context Protocol server with FastMCP"
date: 2025-05-12
draft: false
summary: "This tutorial explores how to create a MCP server with FastMCP and why use it."
---

## Creating Your First MCP Server with FastMCP

The Model Context Protocol (MCP) provides a standardized way to connect large language models (LLMs) to external data and functionalities. With FastMCP, building an MCP server becomes a straightforward task. In this tutorial, we will guide you through the process of creating your first MCP server using FastMCP, along with a client to interact with it.

## What is MCP?

The Model Context Protocol allows you to expose data and functionality to LLM applications securely and in a standardized manner. Here are the key components of MCP:

- **Resources**: These offer read-only access to data, similar to GET endpoints. Resources enable LLMs to query static or dynamic data, like files or database content, enriching the interaction with additional context.
- **Tools**: These transform Python functions into functionalities that LLMs can invoke, akin to POST endpoints. Tools extend the capabilities of LLMs by allowing them to perform actions such as querying databases, calling APIs, or making calculations.
- **Prompts**: Parameterized message templates for LLM interactions. Prompts guide LLMs in generating responses by providing structured templates that can be reused across different contexts.

FastMCP simplifies the implementation of MCP servers by managing boilerplate code and allowing you to focus on building your tools and resources.

### Why FastMCP?

FastMCP offers several advantages:

- **Fast**: High-level interface means faster development.
- **Simple**: Minimal boilerplate required.
- **Pythonic**: Intuitive for Python developers.
- **Complete**: Comprehensive implementation of the MCP specification.

## Creating Your First MCP Server

Let’s dive into the code and create an MCP server in a file named `MCP/server.py`.

### Step 1: Set Up Your MCP Server

```python
from mcp.server.fastmcp import FastMCP

# Create an MCP server for managing explainability in a recommendation engine
mcp = FastMCP("Explainable Recommendation Engine")
```

### Step 2: Define Tools

Tools in FastMCP transform regular Python functions into capabilities that LLMs can invoke during conversations. Here’s how we can define a few tools for our server:

```python
# In-memory store for explanations
explanations = []

@mcp.tool()
def add_explanation(item_id: str, explanation: str) -> str:
    """
    Log a new explanation for a recommended item.
    
    Args:
        item_id (str): Identifier of the recommended item.
        explanation (str): Explanation for why it was recommended.
    
    Returns:
        str: Confirmation message.
    """
    explanations.append((item_id, explanation))
    return f"Added explanation for item '{item_id}'."

@mcp.tool()
def list_explanations() -> str:
    """
    List all explanations for recommended items.
    
    Returns:
        str: Formatted list of explanations.
    """
    if not explanations:
        return "No explanations available."
    return "\n".join(f"{idx + 1}. Item {item_id}: {text}" for idx, (item_id, text) in enumerate(explanations))

@mcp.tool()
def update_explanation(index: int, new_explanation: str) -> str:
    """
    Update an existing explanation by its index (1-based).
    
    Args:
        index (int): The index of the explanation to update.
        new_explanation (str): New explanation text.
    
    Returns:
        str: Confirmation or error message.
    """
    if 0 < index <= len(explanations):
        item_id, _ = explanations[index - 1]
        explanations[index - 1] = (item_id, new_explanation)
        return f"Updated explanation for item '{item_id}'."
    return "Invalid index."

@mcp.tool()
def delete_explanation(index: int) -> str:
    """
    Delete an explanation by its index (1-based).
    
    Args:
        index (int): The index of the explanation to remove.
    
    Returns:
        str: Confirmation or error message.
    """
    if 0 < index <= len(explanations):
        item_id, _ = explanations.pop(index - 1)
        return f"Deleted explanation for item '{item_id}'."
    return "Invalid index."
```

### Step 3: Define Resources and Prompts

Resources provide read-only access to data, while prompts generate parameterized message templates.

#### Resources

Resources are crucial for providing LLMs with access to static and dynamic data.

```python
@mcp.resource("explanations://latest")
def get_latest_explanation() -> str:
    """
    Get the most recently added explanation.
    
    Returns:
        str: The latest explanation or default message.
    """
    return f"Item {explanations[-1][0]}: {explanations[-1][1]}" if explanations else "No explanations yet."
```

#### Prompts

Prompts allow for consistent and reusable message templates, which can greatly enhance the communication between clients and LLMs.

```python
@mcp.prompt()
def explanation_summary_prompt() -> str:
    """
    Generate a summarization prompt for the current list of explanations.
    
    Returns:
        str: AI-ready prompt string, or default message.
    """
    if not explanations:
        return "There are no explanations to summarize."
    formatted = "; ".join(f"{item_id}: {text}" for item_id, text in explanations)
    return f"Summarize the following recommendation rationales: {formatted}"

if __name__ == "__main__":
    mcp.run(transport="stdio")    
```

### Step 4: Run the Server

Finally, you can run your MCP server by adding the following code:

```bash
fastmcp run server.py
```

## Interacting with Your MCP Server

Now that your MCP server is running, let's create a client to interact with it. Save the following code in a file named `MCP/client.py`.

```python
import asyncio
from fastmcp import Client

async def main():
    async with Client("server.py") as client:
        # List the tools exposed by the server
        tools = await client.list_tools()
        print("Available tools:")
        for tool in tools:
            print(f"- {tool.name}: {tool.description.strip().splitlines()[0]}")

        # Example call to add_explanation tool
        print("\nCalling add_explanation...")
        response = await client.call_tool("add_explanation", {
            "item_id": "product_123",
            "explanation": "Frequently bought together with other items in the cart."
        })
        print(f"Response: {response}")

        # Call list_explanations to see the effect
        print("\nCalling list_explanations...")
        response = await client.call_tool("list_explanations", {})
        print(f"Response:\n{response}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Conclusion

In this tutorial, you learned how to create an MCP server using FastMCP and interact with it through a client. This setup allows you to expose tools, resources, and prompts to enhance the capabilities of your LLM applications. You can now expand on this foundation by adding more functionality and exploring the full potential of the Model Context Protocol.