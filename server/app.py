"""
DealRoom FastAPI Server
Thin HTTP wrapper only. Zero business logic. All logic in deal_room/.
"""

import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from models import DealRoomAction
from server.session_pool import DealRoomSessionPool, SESSION_COOKIE_NAME

app = FastAPI(title="DealRoom", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_sessions = DealRoomSessionPool()


def _web_shell_html() -> str:
    return """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>DealRoom Web</title>
        <style>
          :root {
            color-scheme: dark;
          }
          * { box-sizing: border-box; }
          html, body {
            margin: 0;
            height: 100%;
            background: #0b1118;
            overflow: hidden;
          }
          iframe {
            border: 0;
            width: 100%;
            height: 100vh;
            background: #0b1118;
          }
        </style>
      </head>
      <body>
        <iframe id="dealroom-ui-frame" src="/ui/" title="DealRoom Web UI"></iframe>
        <script>
          const search = window.location.search || "";
          const frame = document.getElementById("dealroom-ui-frame");
          if (frame && search) {
            frame.src = "/ui/" + search;
          }
        </script>
      </body>
    </html>
    """


@app.get("/")
async def root():
    return HTMLResponse(_web_shell_html())


@app.get("/web")
async def web_shell():
    return HTMLResponse(_web_shell_html())


@app.get("/web/")
async def web_shell_slash():
    return HTMLResponse(_web_shell_html())


class ResetRequest(BaseModel):
    task_id: Optional[str] = "aligned"
    seed: Optional[int] = 42
    episode_id: Optional[str] = None


def _resolve_session_id(
    request: Request,
    explicit_session_id: Optional[str] = None,
    action: Optional[DealRoomAction] = None,
) -> Optional[str]:
    metadata = action.metadata if action else {}
    return (
        explicit_session_id
        or metadata.get("session_id")
        or metadata.get("episode_id")
        or request.headers.get("x-session-id")
        or request.query_params.get("session_id")
        or request.cookies.get(SESSION_COOKIE_NAME)
    )


def _set_session_cookie(response: Response, session_id: str) -> None:
    response.set_cookie(
        SESSION_COOKIE_NAME,
        session_id,
        max_age=60 * 60 * 6,
        httponly=False,
        samesite="lax",
    )


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
async def reset(
    request: Request,
    response: Response,
    req: ResetRequest = ResetRequest(),
):
    try:
        session_id = _resolve_session_id(request, explicit_session_id=req.episode_id)
        session_id, obs, _state = _sessions.reset(
            seed=req.seed,
            task_id=req.task_id or "aligned",
            session_id=session_id,
        )
        obs.metadata["session_id"] = session_id
        _set_session_cookie(response, session_id)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")


@app.post("/step")
async def step(request: Request, response: Response, action: DealRoomAction):
    try:
        session_id = _resolve_session_id(request, action=action)
        if not session_id or not _sessions.has_session(session_id):
            raise HTTPException(status_code=400, detail="No active session. Call /reset first.")
        obs, reward, done, info, _state = _sessions.step(session_id, action)
        info["session_id"] = session_id
        obs.metadata["session_id"] = session_id
        _set_session_cookie(response, session_id)
        return {
            "observation": obs.model_dump(),
            "reward": reward,
            "done": done,
            "info": info,
        }
    except HTTPException:
        raise
    except KeyError:
        raise HTTPException(status_code=400, detail="No active session. Call /reset first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {e}")


@app.get("/state")
async def state(request: Request, response: Response, session_id: Optional[str] = Query(default=None)):
    try:
        resolved_session_id = _resolve_session_id(request, explicit_session_id=session_id)
        if not resolved_session_id or not _sessions.has_session(resolved_session_id):
            raise HTTPException(status_code=400, detail="No active session. Call /reset first.")
        _set_session_cookie(response, resolved_session_id)
        return _sessions.state(resolved_session_id).model_dump()
    except HTTPException:
        raise
    except KeyError:
        raise HTTPException(status_code=400, detail="No active session. Call /reset first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"State failed: {e}")


def _web_enabled() -> bool:
    return os.getenv("ENABLE_WEB_INTERFACE", "true").lower() == "true"


def _setup_gradio_ui():
    """Setup Gradio UI."""
    global app, gr

    # Try standalone Gradio UI first
    try:
        import gradio as gr
        from server.gradio_standalone import create_dealroom_gradio_app

        _gradio_app = create_dealroom_gradio_app(_sessions)
        app = gr.mount_gradio_app(app, _gradio_app, path="/ui")
        return True
    except ImportError as e:
        print(f"Standalone Gradio not available: {e}")

    # Fall back to OpenEnv Gradio if available
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
        from openenv.core.env_server.web_interface import (
            _extract_action_fields,
            _is_chat_env,
        )
        from server.gradio_custom import (
            DealRoomWebManager,
            build_custom_tab,
            load_metadata,
        )

        _metadata = load_metadata()
        _web_manager = DealRoomWebManager(_sessions, _metadata)
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
            path="/ui",
            theme=OPENENV_GRADIO_THEME,
            css=OPENENV_GRADIO_CSS,
        )
        return True
    except Exception as e:
        print(f"Gradio setup failed: {e}")
        return False


if _web_enabled():
    if not _setup_gradio_ui():

        @app.get("/ui")
        @app.get("/ui/")
        async def web_unavailable():
            return HTMLResponse(
                "<h1>DealRoom Web UI unavailable</h1>"
                "<p>Gradio is not installed. Run: pip install gradio</p>",
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
