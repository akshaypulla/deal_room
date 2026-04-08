"""
DealRoom FastAPI Server
Thin HTTP wrapper only. Zero business logic. All logic in deal_room/.
"""

import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

from models import DealRoomAction
from server.deal_room_environment import DealRoomEnvironment

try:
    import gradio as gr
    from openenv.core.env_server.gradio_theme import (
        OPENENV_GRADIO_CSS,
        OPENENV_GRADIO_THEME,
    )
    from openenv.core.env_server.gradio_ui import (
        build_gradio_app,
        get_gradio_display_title,
    )
    from openenv.core.env_server.web_interface import _extract_action_fields, _is_chat_env
    from server.gradio_custom import DealRoomWebManager, build_custom_tab, load_metadata
except Exception:  # pragma: no cover - graceful local fallback if OpenEnv/Gradio are absent
    gr = None
    OPENENV_GRADIO_CSS = None
    OPENENV_GRADIO_THEME = None
    build_gradio_app = None
    get_gradio_display_title = None
    _extract_action_fields = None
    _is_chat_env = None
    DealRoomWebManager = None
    build_custom_tab = None
    load_metadata = None

app = FastAPI(title="DealRoom", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_env = DealRoomEnvironment()


@app.get("/")
async def root():
    return RedirectResponse(url="/web")


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


def _web_enabled() -> bool:
    return os.getenv("ENABLE_WEB_INTERFACE", "true").lower() == "true"


if (
    _web_enabled()
    and gr is not None
    and build_gradio_app is not None
    and get_gradio_display_title is not None
    and _extract_action_fields is not None
    and _is_chat_env is not None
    and DealRoomWebManager is not None
    and build_custom_tab is not None
    and load_metadata is not None
):
    _metadata = load_metadata()
    _web_manager = DealRoomWebManager(_env, _metadata)
    _action_fields = _extract_action_fields(DealRoomAction)
    _playground = build_gradio_app(
        _web_manager,
        _action_fields,
        _metadata,
        _is_chat_env(DealRoomAction),
        title=_metadata.name,
        quick_start_md=None,
    )
    _custom = build_custom_tab(
        _web_manager,
        _action_fields,
        _metadata,
        _is_chat_env(DealRoomAction),
        _metadata.name,
        None,
    )
    _web_blocks = gr.TabbedInterface(
        [_playground, _custom],
        tab_names=["Playground", "Custom"],
        title=get_gradio_display_title(_metadata),
    )
    app = gr.mount_gradio_app(
        app,
        _web_blocks,
        path="/web",
        theme=OPENENV_GRADIO_THEME,
        css=OPENENV_GRADIO_CSS,
    )
else:
    @app.get("/web")
    async def web_unavailable():
        return HTMLResponse(
            "<h1>DealRoom Web UI unavailable</h1>"
            "<p>The OpenEnv playground is not enabled in this runtime. "
            "Set ENABLE_WEB_INTERFACE=true and ensure Gradio/OpenEnv are installed.</p>",
            status_code=503,
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


def main():
    import uvicorn

    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)
