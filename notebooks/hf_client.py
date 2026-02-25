import asyncio
from fastmcp import Client

async def main():
    async with Client("hf_server.py") as client:
        result = await client.read_resource("hf://models")
        print(result[0].text[:300])

asyncio.run(main())