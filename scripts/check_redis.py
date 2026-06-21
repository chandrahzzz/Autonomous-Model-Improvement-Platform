import asyncio
import redis.asyncio as r


async def check():
    client = r.from_url("redis://:password@localhost:6379/0", decode_responses=True)
    val = await client.get("pipeline:state")
    keys = await client.keys("pipeline:*")
    print("state key:", val)
    print("all pipeline keys:", keys)
    await client.aclose()


asyncio.run(check())
