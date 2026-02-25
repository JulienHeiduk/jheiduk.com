import asyncio
from fastmcp import Client


async def main():
    async with Client("hf_server.py") as client:
        # 1. Trending model catalogue
        result = await client.read_resource("hf://models")
        print("Trending models (first 400 chars):")
        print(result[0].text[:400])

        # 2. Specific model detail via URI template
        result = await client.read_resource("hf://models/google/gemma-2-2b")
        print("\nModel detail — google/gemma-2-2b:")
        print(result[0].text)

        # 3. Trending dataset catalogue
        result = await client.read_resource("hf://datasets")
        print("\nTrending datasets (first 400 chars):")
        print(result[0].text[:400])

        # 4. Specific dataset detail via URI template
        result = await client.read_resource("hf://datasets/rajpurkar/squad")
        print("\nDataset detail — rajpurkar/squad:")
        print(result[0].text)


asyncio.run(main())
