"""Session-scoped DealRoom environment pool."""

from __future__ import annotations

import os
import sys
import time
import uuid
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, Optional, Tuple

_env_path = os.environ.get("DEALROOM_ENV_PATH", "/app/env")
if _env_path not in sys.path:
    sys.path.insert(0, _env_path)

from models import DealRoomAction, DealRoomObservation, DealRoomState

SESSION_COOKIE_NAME = "dealroom_session_id"


@dataclass
class SessionEntry:
    env: Any
    last_access: float


class DealRoomSessionPool:
    """Keeps one environment instance per browser/client session."""

    def __init__(self, max_sessions: int = 128, ttl_seconds: int = 60 * 60 * 6):
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self._sessions: Dict[str, SessionEntry] = {}
        self._lock = Lock()

    def reset(
        self,
        task_id: str,
        seed: Optional[int],
        session_id: Optional[str] = None,
    ) -> Tuple[str, DealRoomObservation, DealRoomState]:
        _base = os.environ.get("DEALROOM_ENV_PATH", "/app/env")
        if _base not in sys.path:
            sys.path.insert(0, _base)

        from deal_room import DealRoomV3

        with self._lock:
            self._prune_locked()
            resolved_session_id = session_id or self._new_session_id()
            entry = self._sessions.get(resolved_session_id)
            if entry is None:
                entry = SessionEntry(env=DealRoomV3(), last_access=time.time())
                self._sessions[resolved_session_id] = entry
            obs = entry.env.reset(seed=seed, task_id=task_id)
            entry.last_access = time.time()
            state = entry.env._state
            if len(self._sessions) > self.max_sessions:
                self._prune_oldest_locked()
            return resolved_session_id, obs, state

    def step(
        self,
        session_id: str,
        action: DealRoomAction,
    ) -> Tuple[DealRoomObservation, float, bool, dict, DealRoomState]:
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                raise KeyError(session_id)
            obs, reward, done, info = entry.env.step(action)
            entry.last_access = time.time()
            return obs, reward, done, info, entry.env._state

    def state(self, session_id: str) -> DealRoomState:
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                raise KeyError(session_id)
            entry.last_access = time.time()
            return entry.env._state

    def get_beliefs(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                return None
            entry.last_access = time.time()
            return entry.env._beliefs

    def has_session(self, session_id: Optional[str]) -> bool:
        if not session_id:
            return False
        with self._lock:
            return session_id in self._sessions

    @staticmethod
    def _new_session_id() -> str:
        return uuid.uuid4().hex[:16]

    def _prune_locked(self) -> None:
        now = time.time()
        expired = [
            session_id
            for session_id, entry in self._sessions.items()
            if now - entry.last_access > self.ttl_seconds
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)

    def _prune_oldest_locked(self) -> None:
        if not self._sessions:
            return
        oldest = min(self._sessions.items(), key=lambda item: item[1].last_access)[0]
        self._sessions.pop(oldest, None)
