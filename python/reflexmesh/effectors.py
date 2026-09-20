"""Opt-in tools with explicit filesystem roots, process templates and endpoints."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import stat
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from .processes import spawn, validate_executable
from .runtime import AdmissionError, EffectReceipt, Runtime, ToolSpec


def _keys(arguments: dict, required: set[str], optional: set[str] = frozenset()) -> None:
    if (
        not isinstance(arguments, dict)
        or set(arguments) - required - optional
        or required - set(arguments)
    ):
        raise AdmissionError("tool arguments do not match the registered schema")


class WorkspaceFiles:
    """Workspace-only UTF-8 tools. Reparse points, device paths and traversal reject.

    Checks precede every operation, but cannot isolate hostile external filesystem
    writers. Deploy in an OS sandbox when the workspace contains adversarial code.
    """

    def __init__(self, root: str | Path, max_bytes: int = 1_048_576):
        self.root = Path(root).resolve(strict=True)
        self.max_bytes = max_bytes

    def path(self, value: str) -> Path:
        if (
            not isinstance(value, str)
            or not value
            or len(value) > 1024
            or "\\" in value
            or ":" in value
            or "\x00" in value
        ):
            raise AdmissionError("path must be a relative portable workspace path")
        relative = PurePosixPath(value)
        if relative.is_absolute() or any(
            part.casefold() in {"..", ".reflexmesh"} for part in relative.parts
        ):
            raise AdmissionError("path is outside the workspace capability")
        current = self.root
        for part in relative.parts:
            if part.rstrip(". ") != part or part.split(".")[0].upper() in {
                "CON",
                "PRN",
                "AUX",
                "NUL",
                *[f"COM{i}" for i in range(10)],
                *[f"LPT{i}" for i in range(10)],
            }:
                raise AdmissionError("reserved or ambiguous path component")
            current = current / part
            if current.is_symlink():
                raise AdmissionError("symlink components are excluded")
            try:
                attributes = current.lstat()
            except FileNotFoundError:
                continue
            if getattr(attributes, "st_file_attributes", 0) & getattr(
                stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024
            ):
                raise AdmissionError("reparse points are excluded")
            if stat.S_ISREG(attributes.st_mode) and attributes.st_nlink > 1:
                raise AdmissionError("hardlinked files require an explicit isolated adapter")
        resolved = current.resolve()
        if not resolved.is_relative_to(self.root) or resolved == self.root:
            raise AdmissionError("path must name a file below the workspace root")
        return resolved

    def canonical(self, value: str) -> str:
        relative = self.path(value).relative_to(self.root).as_posix()
        return relative.casefold() if os.name == "nt" else relative

    def fingerprint(self, resource: str) -> str:
        if not resource.startswith("file:"):
            raise AdmissionError("unknown filesystem resource")
        path = self.path(resource[5:])
        if not path.exists():
            return "absent"
        if not path.is_file() or path.stat().st_size > self.max_bytes:
            raise AdmissionError("tool supports bounded regular files only")
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    def register(self, runtime: Runtime) -> None:
        def validate_read(args: dict) -> dict:
            _keys(args, {"path"})
            return {"path": self.canonical(args["path"])}

        def validate_write(args: dict) -> dict:
            _keys(args, {"path", "content"})
            path = self.canonical(args["path"])
            if (
                not isinstance(args["content"], str)
                or len(args["content"].encode()) > self.max_bytes
            ):
                raise AdmissionError("content exceeds the UTF-8 write capability")
            return {"path": path, "content": args["content"]}

        def read(args: dict, cancelled: threading.Event) -> EffectReceipt:
            path = self.path(args["path"])
            if cancelled.is_set():
                return EffectReceipt("cancelled_verified", {"started": False})
            if not path.is_file():
                return EffectReceipt("failed_verified", {"exists": False, "path": args["path"]})
            with path.open("rb") as stream:
                content = stream.read(self.max_bytes + 1)
            if len(content) > self.max_bytes:
                return EffectReceipt("failed_verified", {"limit_exceeded": True})
            try:
                decoded = content.decode("utf-8")
            except UnicodeDecodeError:
                return EffectReceipt("failed_verified", {"encoding": "not_utf8"})
            return EffectReceipt(
                "completed_verified",
                {
                    "path": args["path"],
                    "content": decoded,
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                },
            )

        def write(args: dict, cancelled: threading.Event) -> EffectReceipt:
            path = self.path(args["path"])
            if not path.parent.is_dir():
                return EffectReceipt("failed_verified", {"parent_missing": True})
            content = args["content"].encode()
            resource = "file:" + args["path"]
            if cancelled.is_set():
                return EffectReceipt("cancelled_verified", {"started": False})
            descriptor, temporary = tempfile.mkstemp(prefix=".reflexmesh-write-", dir=path.parent)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                if cancelled.is_set():
                    return EffectReceipt("cancelled_verified", {"destination_changed": False})
                self.path(args["path"])
                os.replace(temporary, path)
                actual = path.read_bytes()
                expected_hash = hashlib.sha256(content).hexdigest()
                actual_hash = hashlib.sha256(actual).hexdigest()
                return EffectReceipt(
                    "completed_verified" if actual_hash == expected_hash else "effect_unknown",
                    {
                        "path": args["path"],
                        "sha256": actual_hash,
                        "expected_sha256": expected_hash,
                        "bytes": len(actual),
                    },
                    (resource,),
                )
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)

        runtime.register(
            ToolSpec(
                "file.read",
                ("file.read",),
                validate_read,
                lambda a: {"file:" + a["path"]: "read"},
                read,
                self.fingerprint,
            )
        )
        runtime.register(
            ToolSpec(
                "file.write",
                ("file.write",),
                validate_write,
                lambda a: {"file:" + a["path"]: "write"},
                write,
                self.fingerprint,
                allow_write=True,
            )
        )


@dataclass(frozen=True)
class ProcessTemplate:
    name: str
    argv: tuple[str, ...]
    timeout_seconds: float = 30
    output_limit_bytes: int = 65_536
    # Executables must be trusted. Owned Job/process-group lifetime is enforced;
    # privileged launch services and deliberate POSIX session escape require OS isolation.


def register_process(runtime: Runtime, template: ProcessTemplate) -> None:
    """Register one immutable argv, with no shell or model-controlled arguments."""
    if not template.argv or not 0 < template.timeout_seconds <= 3600 or not template.name:
        raise AdmissionError("invalid process template")
    executable = Path(template.argv[0]).resolve(strict=True)
    validate_executable(executable)
    if not executable.is_file():
        raise AdmissionError("executable must be an absolute existing file")
    name = "process." + template.name
    resource = "process:" + template.name
    digest = hashlib.sha256(json.dumps(template.argv).encode()).hexdigest()
    state = {"generation": 0, "running": False, "returncode": None}

    def validate(args: dict) -> dict:
        _keys(args, set())
        return args

    def execute(args: dict, cancelled: threading.Event) -> EffectReceipt:
        if cancelled.is_set():
            return EffectReceipt("cancelled_verified", {"started": False})
        # Preserve platform runtime essentials, never inherit user API credentials.
        env = {
            k: v
            for k, v in os.environ.items()
            if k.upper()
            in {"SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATH", "COMSPEC", "LANG", "LC_ALL"}
        }
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            child = spawn(
                (str(executable), *template.argv[1:]),
                cwd=runtime.workspace,
                stdout=stdout,
                stderr=stderr,
                env=env,
            )
            state.update(generation=state["generation"] + 1, running=True)
            deadline = time.monotonic() + template.timeout_seconds
            interruption = None
            while child.poll() is None:
                if cancelled.wait(0.01) or time.monotonic() > deadline:
                    interruption = "cancelled" if cancelled.is_set() else "timeout"
                    child.terminate()
                    try:
                        child.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=2)
                    break
                if stdout.tell() + stderr.tell() > template.output_limit_bytes * 4:
                    interruption = "output_limit"
                    child.kill()
                    child.wait(timeout=2)
                    break
            state.update(running=False, returncode=child.returncode)
            stdout.seek(0)
            stderr.seek(0)
            evidence = {
                "pid": child.pid,
                "returncode": child.returncode,
                "argv_sha256": digest,
                "stdout": stdout.read(template.output_limit_bytes).decode(errors="replace"),
                "stderr": stderr.read(template.output_limit_bytes).decode(errors="replace"),
                "interruption": interruption,
                "direct_child_terminated": child.poll() is not None,
                "process_containment": child.containment,
                "descendant_termination_verified": child.containment == "windows-job"
                and child.poll() is not None,
            }
            child.close()
            # Termination and nonzero exit do not establish absence of arbitrary effects.
            return EffectReceipt(
                "effect_unknown"
                if interruption
                else ("completed_verified" if child.returncode == 0 else "failed_verified"),
                evidence,
                (resource,),
            )

    runtime.register(
        ToolSpec(
            name,
            (name,),
            validate,
            lambda _: {resource: "write"},
            execute,
            lambda _: hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest(),
            True,
        )
    )


@dataclass(frozen=True)
class HttpEndpoint:
    name: str
    url: str
    method: str = "GET"
    timeout_seconds: float = 10
    max_response_bytes: int = 65_536
    headers: tuple[tuple[str, str], ...] = ()
    allow_loopback_http: bool = False


def register_http(runtime: Runtime, endpoint: HttpEndpoint) -> None:
    """Explicit endpoint capability. Redirects never widen the allowed host/path."""
    parsed = urlsplit(endpoint.url)
    if parsed.username or parsed.password or parsed.fragment or not parsed.hostname:
        raise AdmissionError("endpoint must be an absolute URL without embedded credentials")
    if parsed.scheme != "https" and not (
        parsed.scheme == "http"
        and endpoint.allow_loopback_http
        and parsed.hostname in {"127.0.0.1", "::1", "localhost"}
    ):
        raise AdmissionError("HTTP permitted only for explicitly allowed loopback endpoints")
    if endpoint.method not in {"GET", "POST", "PUT"} or not 0 < endpoint.timeout_seconds <= 120:
        raise AdmissionError("unsupported endpoint contract")
    write = endpoint.method != "GET"
    name = "http." + endpoint.name
    resource = "endpoint:" + endpoint.name
    state = {"generation": 0, "response_sha256": None}

    def validate(args: dict) -> dict:
        _keys(args, {"body", "idempotency_key"} if write else set())
        if write and (
            not isinstance(args["body"], dict)
            or not isinstance(args["idempotency_key"], str)
            or not 1 <= len(args["idempotency_key"]) <= 128
        ):
            raise AdmissionError("write requires an object body and bounded idempotency key")
        if len(json.dumps(args, allow_nan=False).encode()) > 32_768:
            raise AdmissionError("HTTP request exceeds capability limit")
        return args

    def execute(args: dict, cancelled: threading.Event) -> EffectReceipt:
        if cancelled.is_set():
            return EffectReceipt("cancelled_verified", {"started": False})
        connection_type = (
            http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        )
        connection = connection_type(
            parsed.hostname, port=parsed.port, timeout=endpoint.timeout_seconds
        )
        headers = dict(endpoint.headers)
        body = None
        if write:
            body = json.dumps(args["body"], allow_nan=False).encode()
            headers.update(
                {"Content-Type": "application/json", "Idempotency-Key": args["idempotency_key"]}
            )
        try:
            connection.request(
                endpoint.method,
                (parsed.path or "/") + ("?" + parsed.query if parsed.query else ""),
                body,
                headers,
            )
            response = connection.getresponse()
            payload = response.read(endpoint.max_response_bytes + 1)
            state.update(
                generation=state["generation"] + 1,
                response_sha256=hashlib.sha256(payload).hexdigest(),
            )
            evidence = {
                "http_status": response.status,
                "body": payload[: endpoint.max_response_bytes].decode(errors="replace"),
                "response_sha256": state["response_sha256"],
                "redirect_followed": False,
                "response_truncated": len(payload) > endpoint.max_response_bytes,
                "remote_effect_verified": False,
            }
            # HTTP success verifies a response, not an arbitrary remote postcondition.
            status = (
                "effect_unknown"
                if write
                else ("completed_verified" if 200 <= response.status < 300 else "failed_verified")
            )
            return EffectReceipt(status, evidence, (resource,) if write else ())
        except (OSError, http.client.HTTPException) as exc:
            return EffectReceipt(
                "effect_unknown" if write else "failed_verified",
                {"error_type": type(exc).__name__, "remote_effect_verified": False},
            )
        finally:
            connection.close()

    runtime.register(
        ToolSpec(
            name,
            (name,),
            validate,
            lambda _: {resource: "write" if write else "read"},
            execute,
            lambda _: hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest(),
            write,
        )
    )
