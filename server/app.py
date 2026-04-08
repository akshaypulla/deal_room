"""
DealRoom FastAPI Server
Thin HTTP wrapper only. Zero business logic. All logic in deal_room/.
"""

import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import DealRoomAction
from server.deal_room_environment import DealRoomEnvironment

app = FastAPI(title="DealRoom", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_env = DealRoomEnvironment()


class ResetRequest(BaseModel):
    task_id: Optional[str] = "aligned"
    seed: Optional[int] = 42
    episode_id: Optional[str] = None


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "deal-room",
        "tasks": ["aligned", "conflicted", "hostile_acquisition"],
    }


@app.get("/metadata")
async def metadata():
    return {
        "name": "deal-room",
        "version": "1.0.0",
        "tasks": ["aligned", "conflicted", "hostile_acquisition"],
    }


@app.post("/reset")
async def reset(req: ResetRequest = ResetRequest()):
    try:
        obs = _env.reset(seed=req.seed, task_id=req.task_id, episode_id=req.episode_id)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")


@app.post("/step")
async def step(action: DealRoomAction):
    try:
        obs, reward, done, info = _env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward,
            "done": done,
            "info": info,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {e}")


@app.get("/state")
async def state():
    try:
        return _env.state.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"State failed: {e}")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


def main():
    import uvicorn

    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)
