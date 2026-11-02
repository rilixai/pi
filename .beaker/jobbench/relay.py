"""Expose only fixed-model OpenAI inference to an offline task container."""

from __future__ import annotations

import http.client
import json
import socket
import socketserver
import ssl
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Iterator


MAX_REQUEST_BYTES = 64 * 1024 * 1024


def validate_request(path: str, body: bytes, model: str) -> None:
    """Reject alternate models, hosted tools, and provider-side external resources."""
    if path not in ("/v1/responses", "/v1/chat/completions"):
        raise ValueError("Only OpenAI inference endpoints are allowed")
    value = json.loads(body)
    if not isinstance(value, dict) or value.get("model") != model:
        raise ValueError("Candidate execution changed the fixed provider/model")
    if any(value.get(key) for key in ("previous_response_id", "conversation", "background", "store")):
        raise ValueError("Persistent or background provider resources are not allowed")
    if any(tool.get("type") not in ("function", "custom") for tool in value.get("tools", [])):
        raise ValueError("Only local Pi tools are allowed")
    # Tool schemas and function arguments are data; validate message parts only.
    messages = value.get("input", value.get("messages", []))
    if isinstance(messages, str):
        return
    for message in messages:
        if any(tool.get("type") not in ("function", "custom") for tool in message.get("tools", [])):
            raise ValueError("Only local Pi tools are allowed")
        if message.get("type") in ("item_reference", "input_file"):
            raise ValueError("Provider-side files and references are not allowed")
        parts = message.get("content", [])
        if not isinstance(parts, list):
            continue
        for part in parts:
            if part.get("type") in ("input_file", "file"):
                raise ValueError("Provider-side files are not allowed")
            if part.get("type") in ("input_image", "image_url"):
                if part.get("file_id"):
                    raise ValueError("Provider-side image files are not allowed")
                url = part.get("image_url", "")
                if isinstance(url, dict):
                    url = url.get("url", "")
                if not isinstance(url, str) or not url.startswith("data:image/") or ";base64," not in url:
                    raise ValueError("Only inline image data is allowed")


class RelayServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    """Own bounded request threads and close live upstreams at task teardown."""

    daemon_threads = True

    def __init__(self, path: Path, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.slots = threading.BoundedSemaphore(8)
        self.lock = threading.Lock()
        self.active: set[socket.socket] = set()
        self.closing = False
        super().__init__(str(path), RelayHandler)

    def process_request(self, request: socket.socket, client_address: str) -> None:
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        with self.lock:
            if self.closing:
                self.shutdown_request(request)
                self.slots.release()
                return
            self.active.add(request)
        try:
            super().process_request(request, client_address)
        except BaseException:
            with self.lock:
                self.active.discard(request)
            self.slots.release()
            raise

    def process_request_thread(self, request: socket.socket, client_address: str) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            with self.lock:
                self.active.discard(request)
            self.slots.release()

    def handle_error(self, request: socket.socket, client_address: str) -> None:
        # Default socketserver tracebacks could include request data or credentials.
        pass


class RelayHandler(BaseHTTPRequestHandler):
    """Keep the native request and SSE response format across the isolation boundary."""

    server: RelayServer

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_POST(self) -> None:
        self.connection.settimeout(600)
        upstream = None
        upstream_socket = None
        started = False
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if self.headers.get("Transfer-Encoding") or not 0 < length <= MAX_REQUEST_BYTES:
                raise ValueError("A bounded Content-Length is required")
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("Incomplete inference request")
            validate_request(self.path, body, self.server.model)
            # The VM resolves api.openai.com through Beaker's existing provider proxy.
            # Its CA bundle and real credentials stay outside the task container.
            upstream = http.client.HTTPSConnection("api.openai.com", timeout=600, context=ssl.create_default_context())
            upstream.connect()
            upstream_socket = upstream.sock
            with self.server.lock:
                if self.server.closing:
                    raise ConnectionError("Task relay closed")
                self.server.active.add(upstream_socket)
            headers = {
                "Authorization": "Bearer " + self.server.api_key,
                "Content-Type": "application/json", "Accept-Encoding": "identity",
            }
            for name in ("Accept", "OpenAI-Beta", "User-Agent"):
                if self.headers.get(name):
                    headers[name] = self.headers[name]
            upstream.request("POST", self.path, body=body, headers=headers)
            response = upstream.getresponse()
            self.send_response(response.status)
            self.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
            for name in ("Retry-After", "X-Request-Id"):
                if value := response.getheader(name):
                    self.send_header(name, value)
            self.end_headers()
            started = True
            while chunk := response.read1(65536):
                self.wfile.write(chunk)
                self.wfile.flush()
        except (ValueError, TypeError, AttributeError) as error:
            if not started:
                self.send_error(400, str(error).replace(self.server.api_key, "[REDACTED]"))
        except Exception:
            if not started:
                self.send_error(502, "Pi model relay upstream unavailable")
        finally:
            if upstream_socket is not None:
                with self.server.lock:
                    self.server.active.discard(upstream_socket)
            if upstream is not None:
                upstream.close()


@contextmanager
def model_relay(directory: Path, *, api_key: str, model: str) -> Iterator[Path]:
    """Serve one task, then stop its incoming and upstream sockets."""
    directory.mkdir(mode=0o755)
    path = directory / "openai.sock"
    server = RelayServer(path, api_key, model)
    path.chmod(0o666)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    thread.start()
    try:
        yield directory
    finally:
        with server.lock:
            server.closing = True
            for connection in server.active:
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        path.unlink(missing_ok=True)
        directory.rmdir()
