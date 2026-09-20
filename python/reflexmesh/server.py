"""Authenticated local execution service; canonical artifacts remain read-only."""

from __future__ import annotations

import hmac
import json
import secrets
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from starlette.requests import Request


@dataclass
class Run:
    run_id: str
    request: dict
    status: str = "queued"
    events: list[dict] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    cancellation: threading.Event = field(default_factory=threading.Event)
    error: str | None = None
    result: dict | None = None


class RunManager:
    def __init__(
        self,
        checkpoints: str | Path,
        planner: Any = None,
        *,
        capacity: int = 2,
        retention: int = 128,
        policy_factory=None,
    ):
        self.checkpoints, self.planner = Path(checkpoints).resolve(), planner
        self._pool = ThreadPoolExecutor(max_workers=capacity, thread_name_prefix="reflexmesh-run")
        self._slots = threading.BoundedSemaphore(capacity)
        self._runs: dict[str, Run] = {}
        self._retention = retention
        self._lock = threading.RLock()
        self.policy_factory = policy_factory

    def start(self, request: dict) -> Run:
        with self._lock:
            if not self._slots.acquire(blocking=False):
                raise RuntimeError("run capacity exhausted")
            if len(self._runs) >= self._retention:
                for key, old in list(self._runs.items()):
                    if old.status in {"succeeded", "failed", "cancelled"}:
                        del self._runs[key]
                        break
                if len(self._runs) >= self._retention:
                    self._slots.release()
                    raise RuntimeError("retained run capacity exhausted")
            run = Run(uuid4().hex, request)
            self._runs[run.run_id] = run
            try:
                self._pool.submit(self._execute, run)
            except BaseException:
                del self._runs[run.run_id]
                self._slots.release()
                raise
            return run

    def _execute(self, run: Run) -> None:
        from .environments import CaseSpec
        from .evaluation.runner import run_episode
        from .pipeline import make_policy

        policy = None
        try:
            with self._lock:
                run.status = "running"
            factory = self.policy_factory or make_policy
            policy = factory(run.request["method_id"], self.checkpoints, planner=self.planner)
            spec = CaseSpec(
                run.request["case_id"],
                run.request["variant"],
                run.request["seed"],
                "test",
                max_steps=run.request["parameters"].get("max_steps", 16),
            )

            def on_step(event, decision):
                with self._lock:
                    run.events.append(event)
                    run.decisions.append(decision)

            result = run_episode(
                policy,
                spec,
                run.request["parameters"],
                on_step=on_step,
                cancel_event=run.cancellation,
            )
            result["run_id"] = run.run_id
            with self._lock:
                run.result = result
                run.status = "cancelled" if run.cancellation.is_set() else result["status"]
                run.metrics = result["metrics"]
                run.provenance = result["provenance"]
        except Exception as exc:  # noqa: BLE001 - isolate worker failures and retain a typed terminal error
            with self._lock:
                run.status = "cancelled" if run.cancellation.is_set() else "failed"
                # Do not expose local paths, provider messages or credentials over HTTP.
                run.error = type(exc).__name__
        finally:
            try:
                if policy and hasattr(policy, "close"):
                    policy.close()
            finally:
                self._slots.release()

    def snapshot(self, run_id: str, after: int = 0) -> dict:
        with self._lock:
            run = self._runs[run_id]
            return {
                "run_id": run_id,
                "status": run.status,
                "error": run.error,
                "events": [e for e in run.events if e["sequence"] > after],
                "decisions": [d for d in run.decisions if d["sequence"] > after],
                "metrics": dict(run.metrics),
                "provenance": dict(run.provenance),
                "episode": run.result
                if run.status in {"succeeded", "failed", "cancelled"}
                else None,
                "resources": run.result.get("resources", []) if run.result else [],
                "cancellation_requested": run.cancellation.is_set(),
            }

    def cancel(self, run_id: str) -> dict:
        with self._lock:
            run = self._runs[run_id]
            if run.status in {"queued", "running", "cancelling"}:
                run.cancellation.set()
                run.status = "cancelling"
            return {
                "run_id": run_id,
                "status": run.status,
                "cancellation_requested": run.cancellation.is_set(),
                "rollback_claimed": False,
            }

    def close(self):
        with self._lock:
            for run in self._runs.values():
                if run.status in {"queued", "running", "cancelling"}:
                    run.cancellation.set()
        self._pool.shutdown(wait=True, cancel_futures=False)


def create_app(
    *,
    checkpoints: str | Path,
    artifacts: str | Path | None = None,
    token: str | None = None,
    planner: Any = None,
    static: str | Path | None = None,
    policy_factory=None,
    allowed_hosts: tuple[str, ...] = ("localhost", "127.0.0.1", "[::1]"),
):
    """Loopback host checks + bearer auth for every executable request.

    The service executes the registered disposable benchmark workloads. General
    workspace tools use the explicit Python Runtime/Controller SDK or CLI recipe.
    """
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from starlette.middleware.trustedhost import TrustedHostMiddleware

    from .environments import CASES, VARIANTS
    from .policies import METHODS

    auth_token = token or secrets.token_urlsafe(32)
    if len(auth_token) < 24:
        raise ValueError("local service token must be at least 24 characters")
    manager = RunManager(checkpoints, planner, policy_factory=policy_factory)
    artifact_root = Path(artifacts).resolve() if artifacts else None

    @asynccontextmanager
    async def lifespan(app):
        yield
        await __import__("asyncio").to_thread(manager.close)

    app = FastAPI(
        title="Neuraxis local control",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.manager, app.state.local_token = manager, auth_token
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(allowed_hosts))

    @app.middleware("http")
    async def security(request: Request, call_next):
        from fastapi.responses import JSONResponse

        length = request.headers.get("content-length")
        if length and (not length.isdigit() or int(length) > 2_097_152):
            return JSONResponse({"detail": "request body limit"}, status_code=413)
        if request.url.path.startswith("/api/") and request.url.path != "/api/config":
            origin = request.headers.get("origin")
            if origin:
                parsed = urlsplit(origin)
                if parsed.scheme not in {"http", "https"} or parsed.netloc != request.headers.get(
                    "host"
                ):
                    return JSONResponse(
                        {"detail": "same-origin execution required"}, status_code=403
                    )
            authorization = request.headers.get("authorization", "")
            if not hmac.compare_digest(authorization, "Bearer " + auth_token):
                return JSONResponse({"detail": "local bearer token required"}, status_code=401)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = (
            "no-store" if request.url.path.startswith("/api/") else "no-cache"
        )
        return response

    @app.get("/healthz")
    async def health():
        return {"status": "ok", "engine": "reflexmesh", "execution": "local-authenticated"}

    @app.get("/api/config")
    async def config():
        return {
            "schema_version": 1,
            "methods": [{"id": k, "name": v} for k, v in METHODS.items()],
            "cases": [
                {"id": k, "name": v[0], "description": v[2], "variants": list(VARIANTS)}
                for k, v in CASES.items()
            ],
            "capabilities": ["owned-real-software", "native-authority", "replay", "cancellation"],
            "local_auth_required": True,
            "planner_available": planner is not None,
            "checkpoint_available": Path(checkpoints).exists(),
        }

    async def body(request):
        chunks, length = [], 0
        async for chunk in request.stream():
            length += len(chunk)
            if length > 2_097_152:
                raise HTTPException(413, "request body limit")
            chunks.append(chunk)
        raw = b"".join(chunks)
        try:

            def reject_constant(value):
                raise ValueError("nonfinite JSON value: " + value)

            value = json.loads(raw, parse_constant=reject_constant)
        except (ValueError, UnicodeDecodeError):
            raise HTTPException(422, "invalid JSON") from None
        if not isinstance(value, dict):
            raise HTTPException(422, "expected JSON object")
        return value

    @app.post("/api/run")
    async def run(request: Request):
        value = await body(request)
        if set(value) - {"case_id", "method_id", "variant", "seed", "parameters"}:
            raise HTTPException(422, "unknown run fields")
        value = {"variant": "nominal", "seed": 40000, "parameters": {}, **value}
        if (
            value.get("case_id") not in CASES
            or value.get("method_id") not in METHODS
            or value["variant"] not in VARIANTS
        ):
            raise HTTPException(422, "unknown method, case or variant")
        if type(value["seed"]) is not int or not 0 <= value["seed"] <= 2**31 - 1:
            raise HTTPException(422, "seed must be a bounded integer")
        parameters = value["parameters"]
        if not isinstance(parameters, dict) or set(parameters) - {"max_steps"}:
            raise HTTPException(422, "unsupported workload parameters")
        if "max_steps" in parameters and (
            type(parameters["max_steps"]) is not int or not 1 <= parameters["max_steps"] <= 64
        ):
            raise HTTPException(422, "max_steps must be 1..64")
        if value["method_id"] in {"M09", "M10", "M11", "M12"} and planner is None:
            raise HTTPException(409, "this method requires a configured planner")
        try:
            result = manager.start(value)
        except RuntimeError:
            raise HTTPException(429, "run capacity exhausted") from None
        return {"run_id": result.run_id, "status": result.status}

    @app.get("/api/events")
    async def events(run_id: str, after: int = 0):
        if after < 0:
            raise HTTPException(422, "after must be nonnegative")
        try:
            return manager.snapshot(run_id, after)
        except KeyError:
            raise HTTPException(404, "unknown run") from None

    @app.post("/api/cancel/{run_id}")
    async def cancel(run_id: str):
        try:
            return manager.cancel(run_id)
        except KeyError:
            raise HTTPException(404, "unknown run") from None

    @app.post("/api/replay")
    async def replay(request: Request):
        value = await body(request)
        required = {
            "schema_version",
            "run_id",
            "case_id",
            "method_id",
            "events",
            "decisions",
            "metrics",
        }
        if not required <= set(value) or value["schema_version"] != 1:
            raise HTTPException(422, "unsupported episode contract")
        if (
            not isinstance(value["events"], list)
            or not isinstance(value["decisions"], list)
            or max(len(value["events"]), len(value["decisions"])) > 10000
        ):
            raise HTTPException(422, "invalid or excessive trace length")
        return {
            "lane": "imported-replay",
            "executes_effects": False,
            "verified_origin": False,
            "episode": value,
        }

    if artifact_root and artifact_root.is_dir():
        app.mount("/data", StaticFiles(directory=artifact_root), name="artifacts")
    if static and Path(static).is_dir():
        root = Path(static).resolve()
        if (root / "assets").is_dir():
            app.mount("/assets", StaticFiles(directory=root / "assets"), name="assets")

        @app.get("/{path:path}")
        async def spa(path: str):
            candidate = (root / path).resolve()
            if candidate.is_relative_to(root) and candidate.is_file():
                return FileResponse(candidate)
            if path.startswith(("api/", "data/")):
                raise HTTPException(404, "not found")
            return FileResponse(root / "index.html")

    return app
