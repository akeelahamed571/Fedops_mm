import uvicorn
from fastapi import FastAPI
import asyncio
import logging
import os

app = FastAPI()
logger = logging.getLogger("client_manager")

@app.get("/status")
def get_status():
    # Return a dummy status; update this with your actual client status if needed.
    return {"client_online": True, "client_training": False}

async def periodic_status_report():
    while True:
        logger.info("Client Manager: Reporting status...")
        # Optionally, update or send status to a central server.
        await asyncio.sleep(10)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(periodic_status_report())

if __name__ == "__main__":
    # Read the manager port from an environment variable; default to 8003 if not provided.
    port = int(os.environ.get("CLIENT_MANAGER_PORT", 8003))
    uvicorn.run(app, host="0.0.0.0", port=port)
