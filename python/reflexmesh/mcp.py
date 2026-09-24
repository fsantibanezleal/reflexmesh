"""Explicitly scoped MCP 2026-07-28 Streamable HTTP tools.

Discovery metadata and tool output remain untrusted data. A local MCPBinding,
registered through Runtime, supplies capabilities, resources and verification.
The client advertises no optional extensions or elicitation capability.
"""

from __future__ import annotations

import base64
import contextvars
import hashlib
import http.client
import json
import re
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from .runtime import AdmissionError, EffectReceipt, Runtime, ToolSpec, wire

PROTOCOL_VERSION = "2026-07-28"
_PREFIX = "io.modelcontextprotocol/"
_HEADER_TOKEN = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_SCHEMA_BUDGET = contextvars.ContextVar("reflexmesh_mcp_schema_budget", default=None)


class MCPError(RuntimeError):
    """Protocol, transport, or explicit remote JSON-RPC failure."""


class MCPRemoteError(MCPError):
    def __init__(self, code: int, message: str, data: Any = None):
        self.code, self.data = code, data
        super().__init__(f"remote_jsonrpc[{code}]: {message}")


def _header_value(value: str | int | bool) -> str:
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, int):
        if not -(2**53 - 1) <= value <= 2**53 - 1:
            raise AdmissionError("MCP header integer outside exact IEEE754 range")
        text = str(value)
    elif isinstance(value, str):
        text = value
    else:
        raise AdmissionError("MCP header parameters must be primitive string/integer/boolean")
    safe = text == text.strip() and all(32 <= ord(c) <= 126 or c == "\t" for c in text)
    sentinel = text.startswith("=?base64?") and text.endswith("?=")
    return (
        text
        if safe and not sentinel
        else "=?base64?" + base64.b64encode(text.encode()).decode() + "?="
    )


def _schema_validator(schema: dict):
    """2020-12 validation without network resolution, bounded work and regex time."""
    try:
        import regex
        from jsonschema import Draft202012Validator, SchemaError, ValidationError, validators
    except ImportError as exc:
        raise ImportError("MCP requires reflexmesh[mcp] (jsonschema and regex)") from exc
    if not isinstance(schema, dict) or schema.get(
        "$schema", "https://json-schema.org/draft/2020-12/schema"
    ) not in {
        "https://json-schema.org/draft/2020-12/schema",
        "https://json-schema.org/draft/2020-12/schema#",
    }:
        raise AdmissionError("MCP adapter supports JSON Schema 2020-12 only")
    total = 0

    def inspect(value, depth=0):
        nonlocal total
        total += 1
        if total > 2048 or depth > 24:
            raise AdmissionError("remote schema exceeds structural bounds")
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"$ref", "$dynamicRef"} and (
                    not isinstance(child, str) or not child.startswith("#")
                ):
                    raise AdmissionError("external JSON schema references are disabled")
                if key == "pattern" and isinstance(child, str) and len(child) > 1024:
                    raise AdmissionError("remote schema pattern too long")
                if key == "patternProperties":
                    # jsonschema's additional/unevaluated-properties helpers run
                    # stdlib re outside keyword hooks. Reject this profile until
                    # every such path has a bounded regex implementation.
                    raise AdmissionError(
                        "patternProperties is outside the bounded MCP schema profile"
                    )
                inspect(child, depth + 1)
        elif isinstance(value, list):
            for child in value:
                inspect(child, depth + 1)

    inspect(schema)
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise AdmissionError("remote JSON Schema is invalid") from exc

    def pattern(validator, expression, instance, current_schema):
        if isinstance(instance, str):
            try:
                matched = regex.search(expression, instance, timeout=0.02)
            except (TimeoutError, regex.error) as exc:
                raise AdmissionError("remote schema regex exceeded bounds or is invalid") from exc
            if matched is None:
                yield ValidationError("value does not match required pattern")

    mapping = dict(Draft202012Validator.VALIDATORS)
    mapping.update(pattern=pattern)

    def bounded(function):
        def check(*args):
            budget = _SCHEMA_BUDGET.get()
            if budget is not None:
                budget[0] += 1
                if budget[0] > 4096 or time.monotonic() > budget[1]:
                    raise AdmissionError("remote schema validation work limit")
            yield from function(*args)

        return check

    validator_type = validators.extend(
        Draft202012Validator, {k: bounded(v) for k, v in mapping.items()}
    )
    validator = validator_type(schema)

    def validate(value):
        token = _SCHEMA_BUDGET.set([0, time.monotonic() + 0.5])
        try:
            validator.validate(value)
        except (ValidationError, RecursionError) as exc:
            raise AdmissionError(f"MCP schema validation failed: {type(exc).__name__}") from exc
        finally:
            _SCHEMA_BUDGET.reset(token)

    return validate


def _header_paths(schema: dict) -> list[tuple[str, tuple[str, ...]]]:
    paths = []
    names = set()

    def visit(value, path=(), reachable=True):
        if isinstance(value, dict):
            if "x-mcp-header" in value:
                name = value["x-mcp-header"]
                if (
                    not reachable
                    or not path
                    or not isinstance(name, str)
                    or not _HEADER_TOKEN.fullmatch(name)
                ):
                    raise AdmissionError("invalid x-mcp-header annotation")
                kind = value.get("type")
                if (
                    not isinstance(kind, str)
                    or kind not in {"string", "integer", "boolean"}
                    or name.lower() in names
                ):
                    raise AdmissionError("x-mcp-header type/name collision")
                names.add(name.lower())
                paths.append((name, path))
            for key, child in value.items():
                if key == "properties" and isinstance(child, dict):
                    for property_name, property_schema in child.items():
                        visit(property_schema, path + (property_name,), reachable)
                elif isinstance(child, (dict, list)):
                    visit(child, path, False)
        elif isinstance(value, list):
            for child in value:
                visit(child, path, False)

    visit(schema)
    return paths


@dataclass(frozen=True)
class MCPServer:
    server_id: str
    url: str
    allowed_tools: tuple[str, ...]
    timeout_seconds: float = 30
    max_response_bytes: int = 1_048_576
    max_request_bytes: int = 65_536
    headers: tuple[tuple[str, str], ...] = ()
    allow_loopback_http: bool = False


@dataclass(frozen=True)
class MCPBinding:
    """Trusted local effect declaration; remote readOnlyHint cannot populate it."""

    tool: str
    capabilities: tuple[str, ...]
    resources: Callable[[dict[str, Any]], dict[str, str]]
    fingerprint: Callable[[str], str]
    validate: Callable[[dict[str, Any]], dict[str, Any]]
    verifier: Callable[[dict[str, Any], dict[str, Any]], EffectReceipt] | None = None
    read_only: bool = False


class MCPClient:
    def __init__(self, server: MCPServer):
        self.server = server
        self._url = urlsplit(server.url)
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", server.server_id):
            raise AdmissionError("server_id must be a stable local identifier")
        if self._url.username or self._url.password or self._url.fragment or not self._url.hostname:
            raise AdmissionError("MCP URL must have an explicit host and no embedded credentials")
        if self._url.scheme != "https" and not (
            self._url.scheme == "http"
            and server.allow_loopback_http
            and self._url.hostname in {"127.0.0.1", "localhost", "::1"}
        ):
            raise AdmissionError("MCP requires HTTPS or explicitly allowed loopback HTTP")
        if (
            not 0 < server.timeout_seconds <= 120
            or not 256 <= server.max_response_bytes <= 8_388_608
        ):
            raise AdmissionError("invalid MCP transport bounds")
        if not 256 <= server.max_request_bytes <= 1_048_576 or len(server.allowed_tools) > 256:
            raise AdmissionError("invalid MCP request or tool bounds")
        if len(set(server.allowed_tools)) != len(server.allowed_tools):
            raise AdmissionError("duplicate allowed tool name")
        for name, value in server.headers:
            if not _HEADER_TOKEN.fullmatch(name) or any(ord(c) < 32 or ord(c) > 126 for c in value):
                raise AdmissionError("unsafe configured MCP header")
            if name.lower() in {
                "host",
                "content-length",
                "content-type",
                "accept",
            } or name.lower().startswith("mcp-"):
                raise AdmissionError("configured header cannot replace transport metadata")
        self.tools: dict[str, dict] = {}
        self.rejected_tools: dict[str, str] = {}
        self._validators: dict[str, Callable] = {}
        self._output_validators: dict[str, Callable] = {}
        self._header_maps: dict[str, list] = {}
        self.notifications: list[dict] = []
        self.discovery: dict | None = None

    @staticmethod
    def _response(message: Any, request_id: str) -> dict | None:
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            raise MCPError("invalid JSON-RPC response")
        if "method" in message:
            if "id" in message or not str(message["method"]).startswith("notifications/"):
                raise MCPError("unadvertised server request on response stream")
            return None
        if message.get("id") != request_id or ("error" in message) == ("result" in message):
            raise MCPError("response ID or result/error envelope mismatch")
        if "error" in message:
            error = message["error"]
            if (
                not isinstance(error, dict)
                or type(error.get("code")) is not int
                or not isinstance(error.get("message"), str)
            ):
                raise MCPError("malformed JSON-RPC error")
            raise MCPRemoteError(error["code"], error["message"], error.get("data"))
        result = message["result"]
        if not isinstance(result, dict):
            raise MCPError("MCP result must be an object")
        # Absent resultType is complete per the revision's compatibility rule.
        if result.get("resultType", "complete") not in {"complete", "input_required"}:
            raise MCPError("unnegotiated result type")
        return result

    def _rpc(
        self,
        method: str,
        params: dict,
        cancelled: threading.Event | None = None,
        parameter_headers: dict | None = None,
    ) -> dict:
        cancelled = cancelled or threading.Event()
        if cancelled.is_set():
            raise MCPError("cancelled before request")
        request_id = uuid4().hex
        params = {
            **params,
            "_meta": {
                _PREFIX + "protocolVersion": PROTOCOL_VERSION,
                _PREFIX + "clientInfo": {"name": "reflexmesh", "version": "0.1.0"},
                _PREFIX + "clientCapabilities": {},
            },
        }
        payload = wire(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        ).encode()
        if len(payload) > self.server.max_request_bytes:
            raise AdmissionError("MCP request byte bound exceeded")
        headers = {
            **dict(self.server.headers),
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": PROTOCOL_VERSION,
            "Mcp-Method": method,
        }
        if method == "tools/call":
            headers["Mcp-Name"] = _header_value(params["name"])
            headers.update(parameter_headers or {})
        connection_type = (
            http.client.HTTPSConnection
            if self._url.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_type(
            self._url.hostname, self._url.port, timeout=self.server.timeout_seconds
        )
        finished = threading.Event()
        deadline = time.monotonic() + self.server.timeout_seconds
        active_socket = [None]

        def watch():
            while not finished.wait(0.02):
                if cancelled.is_set() or time.monotonic() >= deadline:
                    peer = active_socket[0]
                    if peer is not None:
                        try:
                            peer.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                    connection.close()
                    return

        watcher = threading.Thread(target=watch, daemon=True, name="reflexmesh-mcp-deadline")
        watcher.start()
        try:
            connection.request(
                "POST",
                (self._url.path or "/") + ("?" + self._url.query if self._url.query else ""),
                payload,
                headers,
            )
            active_socket[0] = connection.sock
            if cancelled.is_set() or time.monotonic() >= deadline:
                raise MCPError("MCP request cancelled during connection; effect may be unknown")
            response = connection.getresponse()
            if 300 <= response.status < 400:
                raise MCPError("MCP redirects are not authorized by endpoint scope")
            content_type = response.getheader("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type == "application/json":
                raw = response.read(self.server.max_response_bytes + 1)
                if len(raw) > self.server.max_response_bytes:
                    raise MCPError("MCP response byte bound exceeded")
                result = self._response(json.loads(raw), request_id)
                if result is None:
                    raise MCPError("request received only a notification")
            elif content_type == "text/event-stream":
                total, count = 0, 0
                data = []
                result = None
                while result is None:
                    line = response.readline(min(65536, self.server.max_response_bytes - total + 1))
                    total += len(line)
                    if total > self.server.max_response_bytes or time.monotonic() > deadline:
                        raise MCPError("MCP stream exceeded byte/deadline bound")
                    if cancelled.is_set():
                        raise MCPError("MCP request cancelled; remote effect may be unknown")
                    if not line:
                        raise MCPError("MCP stream ended before final response")
                    if len(line) == 65536 and not line.endswith((b"\n", b"\r")):
                        raise MCPError("MCP stream line bound exceeded")
                    line = line.decode("utf-8").rstrip("\r\n")
                    if line == "" and data:
                        message = json.loads("\n".join(data))
                        data = []
                        count += 1
                        if count > 256:
                            raise MCPError("MCP stream message bound exceeded")
                        result = self._response(message, request_id)
                        if result is None:
                            self.notifications.append(message)
                            self.notifications = self.notifications[-256:]
                    elif line.startswith("data:"):
                        data.append(line[5:].removeprefix(" "))
                    # SSE comments, event IDs and event labels carry no JSON-RPC authority.
            else:
                raise MCPError(
                    f"unsupported MCP response content type/status: {content_type}/{response.status}"
                )
            if response.status >= 400:
                raise MCPError(f"HTTP {response.status} did not contain a JSON-RPC error")
            return result
        except (
            OSError,
            http.client.HTTPException,
            json.JSONDecodeError,
            UnicodeDecodeError,
        ) as exc:
            raise MCPError(
                f"MCP transport failed: {type(exc).__name__}; effect may be unknown"
            ) from exc
        finally:
            finished.set()
            connection.close()
            watcher.join(timeout=0.1)

    def discover(self) -> dict:
        result = self._rpc("server/discover", {})
        versions = result.get("supportedVersions")
        if not isinstance(versions, list) or PROTOCOL_VERSION not in versions:
            raise MCPError("server does not advertise supported modern protocol version")
        if (
            not isinstance(result.get("capabilities"), dict)
            or "tools" not in result["capabilities"]
        ):
            raise MCPError("server does not advertise tools capability")
        self.discovery = result
        return result

    def list_tools(self) -> list[dict]:
        if self.discovery is None:
            self.discover()
        cursor, seen = None, set()
        descriptors, validators, outputs, header_maps = {}, {}, {}, {}
        rejected = {}
        for _ in range(16):
            result = self._rpc("tools/list", {"cursor": cursor} if cursor else {})
            if result.get("resultType", "complete") != "complete" or not isinstance(
                result.get("tools"), list
            ):
                raise MCPError("invalid tools/list result")
            for tool in result["tools"]:
                if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
                    raise MCPError("invalid tool descriptor")
                name = tool["name"]
                if name not in self.server.allowed_tools:
                    continue
                if name in descriptors:
                    raise MCPError("duplicate tool name across discovery pages")
                try:
                    validators[name] = _schema_validator(tool["inputSchema"])
                    header_maps[name] = _header_paths(tool["inputSchema"])
                    if "outputSchema" in tool:
                        outputs[name] = _schema_validator(tool["outputSchema"])
                    descriptors[name] = json.loads(wire(tool))
                except (KeyError, ValueError, AdmissionError) as exc:
                    rejected[name] = str(exc)
            cursor = result.get("nextCursor")
            if cursor is None:
                break
            if not isinstance(cursor, str) or cursor in seen or len(cursor) > 4096:
                raise MCPError("invalid/cyclic MCP pagination cursor")
            seen.add(cursor)
        else:
            raise MCPError("MCP tool pagination bound exceeded")
        self.tools, self._validators, self._output_validators = descriptors, validators, outputs
        self._header_maps, self.rejected_tools = header_maps, rejected
        return list(descriptors.values())

    def validate(self, tool: str, arguments: dict) -> dict:
        if tool not in self.server.allowed_tools or tool not in self.tools:
            raise AdmissionError("MCP tool is not explicitly allowed and discovered")
        if (
            not isinstance(arguments, dict)
            or len(wire(arguments).encode()) > self.server.max_request_bytes
        ):
            raise AdmissionError("MCP arguments must be a bounded object")
        self._validators[tool](arguments)
        return json.loads(wire(arguments))

    def _invoke(self, tool: str, arguments: dict, cancelled: threading.Event) -> dict:
        arguments = self.validate(tool, arguments)
        headers = {}
        for name, path in self._header_maps[tool]:
            value = arguments
            for component in path:
                if not isinstance(value, dict) or component not in value:
                    value = None
                    break
                value = value[component]
            if value is not None:
                headers["Mcp-Param-" + name] = _header_value(value)
        result = self._rpc("tools/call", {"name": tool, "arguments": arguments}, cancelled, headers)
        if result.get("resultType", "complete") == "input_required":
            raise MCPError("server requested unadvertised client input; no automatic retry")
        if not isinstance(result.get("content"), list):
            raise MCPError("tool result content must be a list")
        if tool in self._output_validators and not result.get("isError", False):
            self._output_validators[tool](result.get("structuredContent"))
        return result

    def register(self, runtime: Runtime, binding: MCPBinding) -> str:
        if not self.tools:
            self.list_tools()
        if binding.tool not in self.tools:
            raise AdmissionError("tool is absent or its descriptor was rejected")
        pinned_descriptor = hashlib.sha256(wire(self.tools[binding.tool]).encode()).hexdigest()
        local_name = f"mcp.{self.server.server_id}.{binding.tool}"

        def validate(arguments):
            clean = binding.validate(self.validate(binding.tool, arguments))
            accesses = binding.resources(clean)
            if binding.read_only and any(access != "read" for access in accesses.values()):
                raise AdmissionError("read-only local MCP binding cannot declare writes")
            return clean

        def execute(arguments, cancelled):
            if cancelled.is_set():
                return EffectReceipt("cancelled_verified", {"request_sent": False})
            current = hashlib.sha256(wire(self.tools.get(binding.tool)).encode()).hexdigest()
            if current != pinned_descriptor:
                return EffectReceipt(
                    "failed_verified", {"request_sent": False, "schema_changed": True}
                )
            result = self._invoke(binding.tool, arguments, cancelled)
            evidence = {
                "mcp_result": result,
                "server_id": self.server.server_id,
                "protocol_version": PROTOCOL_VERSION,
                "content_trust": "untrusted_tool_output",
            }
            if binding.verifier:
                verified = binding.verifier(arguments, result)
                return EffectReceipt(
                    verified.status,
                    {**evidence, "verification": verified.evidence},
                    verified.changed_resources,
                )
            if binding.read_only:
                return EffectReceipt(
                    "failed_verified" if result.get("isError") else "completed_verified",
                    {**evidence, "verified_property": "bounded MCP response received"},
                )
            return EffectReceipt("effect_unknown", {**evidence, "remote_effect_verified": False})

        runtime.register(
            ToolSpec(
                local_name,
                binding.capabilities,
                validate,
                binding.resources,
                execute,
                binding.fingerprint,
                allow_write=not binding.read_only,
            )
        )
        return local_name
