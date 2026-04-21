"""
DealRoom v3 — MiniMax LLM Client.

Single API: MiniMax M2.5 via curl, Anthropic-compatible /v1/messages endpoint.
No fallback. No JSON scoring.
"""

import os
import sys
import time
import json
import subprocess
import threading
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import atexit


MAX_TOKENS = {
    "stakeholder_response": 200,
    "deliberation_summary": 220,
}
INTERACTIVE_DEBUG_ENV = "DEALROOM_LLM_INTERACTIVE"


class LLMErrorType(Enum):
    NETWORK_TIMEOUT = "network_timeout"
    NETWORK_UNREACHABLE = "network_unreachable"
    CONNECTION_RESET = "connection_reset"
    DNS_FAILURE = "dns_failure"
    RATE_LIMIT_429 = "rate_limit_429"
    QUOTA_EXCEEDED = "quota_exceeded"
    TOKENS_EXCEEDED = "tokens_exceeded"
    AUTH_INVALID_KEY = "auth_invalid_key"
    AUTH_EXPIRED = "auth_expired"
    SERVER_500 = "server_500"
    SERVER_5XX = "server_5xx"
    SERVER_OVERLOADED = "server_overloaded"
    EMPTY_RESPONSE = "empty_response"
    CONTENT_FILTER = "content_filter"
    UNKNOWN = "unknown"


@dataclass
class LLMError:
    error_type: LLMErrorType
    message: str
    api: str = "unknown"
    status_code: Optional[int] = None
    retry_after: Optional[float] = None
    raw_exception: Optional[Exception] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def is_auto_recoverable(self) -> bool:
        return self.error_type in {
            LLMErrorType.NETWORK_TIMEOUT,
            LLMErrorType.NETWORK_UNREACHABLE,
            LLMErrorType.CONNECTION_RESET,
            LLMErrorType.DNS_FAILURE,
            LLMErrorType.SERVER_500,
            LLMErrorType.SERVER_5XX,
            LLMErrorType.SERVER_OVERLOADED,
            LLMErrorType.EMPTY_RESPONSE,
        }

    def is_rate_limit(self) -> bool:
        return self.error_type in {
            LLMErrorType.RATE_LIMIT_429,
            LLMErrorType.QUOTA_EXCEEDED,
            LLMErrorType.TOKENS_EXCEEDED,
        }

    def is_auth_error(self) -> bool:
        return self.error_type in {
            LLMErrorType.AUTH_INVALID_KEY,
            LLMErrorType.AUTH_EXPIRED,
        }

    def requires_user_intervention(self) -> bool:
        return self.is_auth_error() or self.error_type == LLMErrorType.QUOTA_EXCEEDED


def classify_error(
    exception: Exception, status_code: Optional[int] = None, api: str = "unknown"
) -> LLMError:
    msg = str(exception).lower()
    raw = str(exception)

    retry_after = None
    if hasattr(exception, "response") and exception.response is not None:
        h = exception.response.headers.get("retry-after")
        if h:
            try:
                retry_after = float(h)
            except ValueError:
                retry_after = 60.0
        if status_code is None:
            status_code = exception.response.status_code

    def make(etype, message):
        return LLMError(
            error_type=etype,
            message=message,
            api=api,
            status_code=status_code,
            retry_after=retry_after,
            raw_exception=exception,
        )

    if status_code in (401, 403):
        return make(
            LLMErrorType.AUTH_INVALID_KEY,
            f"Authentication failed (HTTP {status_code}). Check your API key.",
        )
    if status_code == 429:
        if any(
            k in msg for k in ["quota", "billing", "exceeded your", "limit exceeded"]
        ):
            return make(
                LLMErrorType.QUOTA_EXCEEDED, f"API quota/billing limit exceeded. {raw}"
            )
        return make(LLMErrorType.RATE_LIMIT_429, f"Rate limit hit (HTTP 429). {raw}")
    if status_code == 500:
        return make(
            LLMErrorType.SERVER_500, f"Server internal error (HTTP 500). Will retry."
        )
    if status_code in (502, 503, 504):
        return make(
            LLMErrorType.SERVER_5XX,
            f"Server unavailable (HTTP {status_code}). Will retry.",
        )

    if any(k in msg for k in ["timeout", "timed out", "read timed out"]):
        return make(LLMErrorType.NETWORK_TIMEOUT, "Request timed out. Will retry.")
    if any(
        k in msg
        for k in [
            "connection refused",
            "connection reset",
            "connection aborted",
            "remote end closed",
        ]
    ):
        return make(LLMErrorType.CONNECTION_RESET, "Connection reset. Will retry.")
    if any(
        k in msg
        for k in [
            "name or service not known",
            "failed to resolve",
            "getaddrinfo failed",
            "name resolution",
        ]
    ):
        return make(LLMErrorType.DNS_FAILURE, "DNS resolution failed. Check network.")
    if any(
        k in msg for k in ["network is unreachable", "no route to host", "enetunreach"]
    ):
        return make(
            LLMErrorType.NETWORK_UNREACHABLE, "Network unreachable. Check connection."
        )
    if any(k in msg for k in ["overloaded", "capacity", "too many requests"]):
        return make(LLMErrorType.SERVER_OVERLOADED, "Server overloaded. Will retry.")
    if any(k in msg for k in ["context length", "maximum context", "too long"]):
        return make(LLMErrorType.TOKENS_EXCEEDED, f"Token limit exceeded: {raw}")
    if any(k in msg for k in ["content policy", "content filter", "safety", "flagged"]):
        return make(
            LLMErrorType.CONTENT_FILTER, "Content filtered. Prompt may need adjustment."
        )
    if not msg or msg in ("", "none"):
        return make(LLMErrorType.EMPTY_RESPONSE, "Empty response from LLM.")

    return make(LLMErrorType.UNKNOWN, f"Unknown error: {raw}")


def get_minimax_client() -> Tuple[Any, str]:
    key = os.environ.get("MINIMAX_API_KEY")
    model = os.environ.get("MINIMAX_MODEL", "MiniMax-Text-01")
    if not key:
        return None, model
    from openai import OpenAI

    base_url = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.chat/v1")
    return OpenAI(api_key=key, base_url=base_url), model


@dataclass
class RetryPolicy:
    max_auto_retries: int = 3
    base_backoff_sec: float = 1.0
    max_backoff_sec: float = 30.0
    backoff_factor: float = 2.0
    jitter_fraction: float = 0.2
    rate_limit_default_wait: float = 60.0
    rate_limit_max_wait: float = 300.0

    def compute_backoff(self, attempt: int) -> float:
        import random

        base = self.base_backoff_sec * (self.backoff_factor**attempt)
        capped = min(base, self.max_backoff_sec)
        jitter = capped * self.jitter_fraction * (2 * random.random() - 1)
        return max(0.1, capped + jitter)


DEFAULT_POLICY = RetryPolicy()


class LLMCallStats:
    def __init__(self):
        self.calls: Dict[str, int] = {"minimax": 0}
        self.successes: Dict[str, int] = {"minimax": 0}
        self.auto_retried: int = 0
        self.interventions: int = 0
        self.skipped: int = 0
        self.errors: Dict[LLMErrorType, int] = {}
        self._lock = threading.Lock()

    def record(self, api, success=False, retry=False, skip=False, error=None):
        with self._lock:
            self.calls[api] = self.calls.get(api, 0) + 1
            if success:
                self.successes[api] = self.successes.get(api, 0) + 1
            if retry:
                self.auto_retried += 1
            if skip:
                self.skipped += 1
            if error:
                self.errors[error] = self.errors.get(error, 0) + 1

    def record_intervention(self):
        with self._lock:
            self.interventions += 1

    def print_summary(self):
        print(f"\n{'=' * 52}")
        print("  DealRoom v3 — LLM Call Summary")
        print(f"{'─' * 52}")
        total = sum(self.calls.values())
        succ = sum(self.successes.values())
        print(f"  MiniMax calls:       {self.calls.get('minimax', 0):>4}")
        print(f"  Total:               {total:>4}   (success: {succ})")
        if self.auto_retried:
            print(f"  Auto-retried:        {self.auto_retried:>4}")
        if self.interventions:
            print(f"  User interventions:  {self.interventions:>4}")
        if self.skipped:
            print(f"  Skipped:             {self.skipped:>4}")
        if self.errors:
            print(f"  Errors:")
            for et, n in sorted(self.errors.items(), key=lambda x: -x[1]):
                print(f"    {et.value:<28} {n}")
        print(f"{'=' * 52}\n")


STATS = LLMCallStats()
atexit.register(STATS.print_summary)


BOLD = "\033[1m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
GREEN = "\033[92m"
DIM = "\033[2m"
RESET = "\033[0m"


def _interactive_pause(error: LLMError, context: str, allow_skip: bool = True) -> str:
    print(f"\n{'=' * 62}")
    print(f"{RED}{BOLD}  LLM CALL FAILED — MANUAL INTERVENTION REQUIRED{RESET}")
    print(f"{'=' * 62}")
    print(f"  {BOLD}API:{RESET}         {error.api}")
    print(f"  {BOLD}Error:{RESET}       {error.error_type.value}")
    print(f"  {BOLD}Message:{RESET}     {error.message}")
    print(f"  {BOLD}Context:{RESET}     {context}")
    print(f"  {BOLD}Time:{RESET}        {error.timestamp.strftime('%H:%M:%S')}")
    if error.status_code:
        print(f"  {BOLD}HTTP status:{RESET} {error.status_code}")
    print(f"{'─' * 62}")

    if error.is_auth_error():
        print(
            f"{YELLOW}  Authentication failed. Fix your API key, then press c.{RESET}"
        )
        print(f"  Current: {_get_key_source('MINIMAX_API_KEY')}")
        print(f"  Fix:     export MINIMAX_API_KEY=your_new_key\n")
    elif error.is_rate_limit():
        print(f"{YELLOW}  Rate/quota limit hit.{RESET}")
        if error.retry_after:
            print(f"  API suggested wait: {error.retry_after:.0f}s")
        print()
    elif error.error_type in (
        LLMErrorType.NETWORK_UNREACHABLE,
        LLMErrorType.DNS_FAILURE,
    ):
        print(f"{YELLOW}  Network issue. Check internet/VPN/firewall.{RESET}\n")

    print(f"{CYAN}{BOLD}  Options:{RESET}")
    print(f"    {GREEN}c{RESET}        continue — I fixed the issue, retry now")
    print(f"    {GREEN}w <N>{RESET}    wait N seconds then retry  (e.g.  w 60)")
    print(f"    {GREEN}r{RESET}        retry immediately")
    if allow_skip:
        print(f"    {GREEN}s{RESET}        skip this call (caller uses fallback/empty)")
    print(f"    {GREEN}e{RESET}        exit the program")
    print(f"{'─' * 62}")

    while True:
        try:
            choice = input(f"  {BOLD}Your choice: {RESET}").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{RED}Interrupted — exiting.{RESET}")
            sys.exit(1)

        if not choice:
            continue
        if choice in ("c", "continue", "r", "retry"):
            print(f"  {GREEN}Resuming...{RESET}\n")
            return "retry"
        if choice.startswith("w"):
            parts = choice.split()
            secs = 60
            if len(parts) == 2:
                try:
                    secs = int(parts[1])
                except ValueError:
                    pass
            print(f"  {CYAN}Waiting {secs}s...{RESET}", end="", flush=True)
            for rem in range(secs, 0, -10):
                time.sleep(min(10, rem))
                print(f" {rem}s", end="", flush=True)
            print(f"  {GREEN}Done. Retrying...{RESET}\n")
            return "retry"
        if choice == "s" and allow_skip:
            print(f"  {YELLOW}Skipped. Caller uses fallback.{RESET}\n")
            return "skip"
        if choice in ("e", "exit", "quit", "q"):
            print(f"  {RED}Exiting.{RESET}")
            sys.exit(0)

        allowed = "c, w <N>, r, s, e" if allow_skip else "c, w <N>, r, e"
        print(f"  Unknown '{choice}'. Enter: {allowed}")


def _get_key_source(var: str) -> str:
    val = os.environ.get(var, "")
    if val:
        return f"{var}=***{val[-4:]}"
    return f"{var}=(not set)"


def _print_auto_retry(error, context, attempt, backoff, max_retries):
    ts = datetime.now().strftime("%H:%M:%S")
    print(
        f"{DIM}  [{ts}] [{error.api}] auto-retry {attempt + 1}/{max_retries} "
        f"for [{context}] — wait {backoff:.1f}s ({error.error_type.value}){RESET}",
        flush=True,
    )


def _print_rate_wait(secs, context, api):
    ts = datetime.now().strftime("%H:%M:%S")
    resumes = (datetime.now() + timedelta(seconds=secs)).strftime("%H:%M:%S")
    print(
        f"\n  {YELLOW}[{ts}] [{api}] rate-limit for [{context}] — "
        f"waiting {secs:.0f}s (resumes {resumes}){RESET}",
        flush=True,
    )


def _countdown_sleep(seconds: float):
    elapsed = 0
    interval = min(10, seconds)
    while elapsed < seconds:
        chunk = min(interval, seconds - elapsed)
        time.sleep(chunk)
        elapsed += chunk
        remaining = seconds - elapsed
        if remaining > 1:
            print(f"  ...{remaining:.0f}s", end="\r", flush=True)
    print(" " * 30, end="\r")


def _strip_json_fences(text: str) -> str:
    s = text.strip()
    for fence in ("```json", "```"):
        if s.startswith(fence):
            s = s[len(fence) :]
    for fence in ("```",):
        if s.endswith(fence):
            s = s[: -len(fence)]
    return s.strip()


def llm_call_text(
    prompt: str,
    call_type: str,
    temperature: float,
    context: str = "",
    allow_skip: bool = False,
    policy: RetryPolicy = DEFAULT_POLICY,
    timeout: float = 30.0,
) -> Optional[str]:
    if not os.environ.get("MINIMAX_API_KEY"):
        STATS.record("minimax", skip=True)
        return None

    max_tokens_val = MAX_TOKENS[call_type]
    attempt = 0
    interactive = os.environ.get(INTERACTIVE_DEBUG_ENV, "").lower() in {
        "1",
        "true",
        "yes",
    }

    while True:
        key = os.environ.get("MINIMAX_API_KEY", "")
        base_url = os.environ.get(
            "MINIMAX_BASE_URL", "https://api.minimax.io/anthropic/v1"
        )
        model = os.environ.get("MINIMAX_MODEL", "MiniMax-M2.5")

        STATS.record("minimax")
        try:
            payload = json.dumps(
                {
                    "model": model,
                    "max_tokens": max_tokens_val,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}],
                }
            )
            result = subprocess.run(
                [
                    "curl",
                    "-s",
                    "-X",
                    "POST",
                    f"{base_url}/messages",
                    "-H",
                    f"Authorization: Bearer {key}",
                    "-H",
                    "Content-Type: application/json",
                    "-H",
                    "anthropic-version: 2023-06-01",
                    "-d",
                    payload,
                    "--max-time",
                    str(int(timeout)),
                ],
                capture_output=True,
                text=True,
                timeout=int(timeout) + 5,
            )

            if result.returncode != 0:
                stderr = result.stderr.strip()
                if any(k in stderr for k in ["timeout", "timed out"]):
                    raise TimeoutError(f"curl timed out: {stderr}")
                raise RuntimeError(f"curl failed (rc={result.returncode}): {stderr}")

            data = json.loads(result.stdout)
            content_blocks = data.get("content", [])
            text_parts = []
            for block in content_blocks:
                if block.get("type") == "text" and block.get("text", "").strip():
                    text_parts.append(block["text"])
            text = " ".join(text_parts) if text_parts else ""

            if not text or not text.strip():
                raise ValueError("Empty response")
            STATS.record("minimax", success=True)
            return text.strip()

        except Exception as raw_exc:
            sc = getattr(raw_exc, "status_code", None)
            if (
                sc is None
                and hasattr(raw_exc, "response")
                and raw_exc.response is not None
            ):
                sc = raw_exc.response.status_code
            err = classify_error(raw_exc, sc, api="minimax")

            print(
                f"{DIM}  [{datetime.now().strftime('%H:%M:%S')}] "
                f"[minimax/{call_type}] {err.error_type.value}{RESET}",
                flush=True,
            )
            STATS.record("minimax", error=err.error_type)

            if err.is_auth_error():
                if not interactive:
                    STATS.record("minimax", skip=True)
                    return None
                action = _interactive_pause(err, context, allow_skip)
                STATS.record_intervention()
                if action == "skip":
                    STATS.record("minimax", skip=True)
                    return None
                continue

            if err.is_rate_limit():
                if err.error_type == LLMErrorType.QUOTA_EXCEEDED:
                    if not interactive:
                        STATS.record("minimax", skip=True)
                        return None
                    action = _interactive_pause(err, context, allow_skip)
                    STATS.record_intervention()
                    if action == "skip":
                        STATS.record("minimax", skip=True)
                        return None
                    continue
                wait = min(
                    err.retry_after or policy.rate_limit_default_wait,
                    policy.rate_limit_max_wait,
                )
                _print_rate_wait(wait, context, "minimax")
                _countdown_sleep(wait)
                STATS.record("minimax", retry=True)
                attempt += 1
                continue

            if err.is_auto_recoverable() and attempt < policy.max_auto_retries:
                backoff = policy.compute_backoff(attempt)
                _print_auto_retry(
                    err, context, attempt, backoff, policy.max_auto_retries
                )
                time.sleep(backoff)
                STATS.record("minimax", retry=True)
                attempt += 1
                continue

            if not interactive:
                STATS.record("minimax", skip=True)
                return None

            action = _interactive_pause(err, context, allow_skip)
            STATS.record_intervention()
            if action == "skip":
                STATS.record("minimax", skip=True)
                return None
            attempt = 0
            continue


def generate_stakeholder_response(prompt: str, context: str = "") -> Optional[str]:
    return llm_call_text(
        prompt=prompt,
        call_type="stakeholder_response",
        temperature=0.7,
        context=context or "stakeholder_response",
        allow_skip=True,
    )


def generate_deliberation_summary(
    prompt: str, context: str = "", timeout: float = 5.0
) -> str:
    result = llm_call_text(
        prompt=prompt,
        call_type="deliberation_summary",
        temperature=0.8,
        context=context or "deliberation_summary",
        allow_skip=True,
        timeout=timeout,
    )
    return result or ""


def validate_api_keys():
    return bool(os.environ.get("MINIMAX_API_KEY"))


USAGE = """
DealRoom v3 — API Configuration

API:   MiniMax M2.5

REQUIRED:
  export MINIMAX_API_KEY=your_minimax_key

OPTIONAL OVERRIDES:
  export MINIMAX_BASE_URL=https://api.minimax.io/anthropic/v1  (default)
  export MINIMAX_MODEL=MiniMax-M2.5                           (default)

CALL ROUTING:
  generate_stakeholder_response() -> MiniMax (text output)
  generate_deliberation_summary() -> MiniMax (text output)
"""


if __name__ == "__main__":
    print(USAGE)
