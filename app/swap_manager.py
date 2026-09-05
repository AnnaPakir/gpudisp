from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import string
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

try:
    import pynvml
except Exception:  # pragma: no cover - optional runtime dependency
    pynvml = None


LOGGER = logging.getLogger("gpudisp.swap")
PRIORITY = {"low": 10, "normal": 50, "medium": 50, "high": 100}
HEADER_DENYLIST = {"host", "content-length"}


@dataclass
class ModelSpec:
    name: str
    aliases: list[str]
    endpoints: set[str]
    estimated_vram_mb: int
    priority: int
    priority_name: str
    pre_warm: bool
    idle_timeout_sec: int
    startup_timeout_sec: int
    request_timeout_sec: int
    health_path: str
    cmd: str
    host: str
    port: int


@dataclass
class ModelState:
    spec: ModelSpec
    process: asyncio.subprocess.Process | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    active_requests: int = 0
    last_used_monotonic: float = field(default_factory=time.monotonic)

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.returncode is None


class VramProbe:
    def __init__(self) -> None:
        self.available = False
        if pynvml is None:
            return
        try:
            pynvml.nvmlInit()
            self.available = True
        except Exception as exc:
            LOGGER.warning("NVML is not available, VRAM checks disabled: %s", exc)

    def free_mb(self) -> int | None:
        if not self.available:
            return None
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return int(info.free / 1024 / 1024)
        except Exception as exc:
            LOGGER.warning("Could not read free VRAM: %s", exc)
            return None


class SwapManager:
    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.raw_config = self._load_config(config_path)
        manager_cfg = self.raw_config.get("manager", {})
        self.host = str(manager_cfg.get("host", "0.0.0.0"))
        self.port = int(manager_cfg.get("port", 9000))
        self.backend_host = str(manager_cfg.get("backend_host", "127.0.0.1"))
        self.backend_base_port = int(manager_cfg.get("backend_base_port", 9100))
        self.default_startup_timeout = int(manager_cfg.get("startup_timeout_sec", 240))
        self.default_request_timeout = int(manager_cfg.get("request_timeout_sec", 900))
        self.idle_check_interval = int(manager_cfg.get("idle_check_interval_sec", 5))
        self.prewarm_check_interval = int(manager_cfg.get("prewarm_check_interval_sec", 10))
        self.vram_guard_mb = int(manager_cfg.get("vram_guard_mb", 768))

        self.vram = VramProbe()
        self.models: dict[str, ModelState] = {}
        self.aliases: dict[str, str] = {}
        self._load_models()
        self.app = self._make_app()

    @staticmethod
    def _load_config(path: str) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if "models" not in data:
            raise RuntimeError("swap config must contain a 'models' section")
        return data

    def _load_models(self) -> None:
        for index, (name, cfg) in enumerate(self.raw_config["models"].items()):
            priority_name = str(cfg.get("priority", "normal")).lower()
            port = int(cfg.get("port", self.backend_base_port + index))
            spec = ModelSpec(
                name=name,
                aliases=[str(alias) for alias in cfg.get("aliases", [])],
                endpoints={str(endpoint) for endpoint in cfg.get("endpoints", [])},
                estimated_vram_mb=int(cfg.get("estimated_vram_mb", 0)),
                priority=PRIORITY.get(priority_name, 50),
                priority_name=priority_name,
                pre_warm=bool(cfg.get("pre_warm", False)),
                idle_timeout_sec=int(cfg.get("idle_timeout_sec", cfg.get("idle_timeout", 0))),
                startup_timeout_sec=int(cfg.get("startup_timeout_sec", self.default_startup_timeout)),
                request_timeout_sec=int(cfg.get("request_timeout_sec", self.default_request_timeout)),
                health_path=str(cfg.get("health_path", "/health")),
                cmd=str(cfg["cmd"]),
                host=str(cfg.get("host", self.backend_host)),
                port=port,
            )
            self.models[name] = ModelState(spec=spec)
            for alias in {name, f"openai/{name}", *spec.aliases}:
                self.aliases[self._normalize_model(alias)] = name

    @staticmethod
    def _normalize_model(model_name: str | None) -> str:
        model = (model_name or "").strip()
        if model.startswith("openai/"):
            model = model.removeprefix("openai/")
        return model

    def _make_app(self) -> FastAPI:
        app = FastAPI(title="GPU Dispatch Swap Manager", version="0.1.0")

        @app.on_event("startup")
        async def _startup() -> None:
            asyncio.create_task(self._idle_reaper())
            asyncio.create_task(self._prewarm_loop())

        @app.get("/")
        @app.get("/health")
        async def _health() -> dict[str, Any]:
            return self._status_payload()

        @app.get("/v1/models")
        async def _models() -> dict[str, Any]:
            return {
                "object": "list",
                "data": [
                    {
                        "id": state.spec.name,
                        "object": "model",
                        "owned_by": "gpudisp",
                        "running": state.running,
                        "priority": state.spec.priority_name,
                    }
                    for state in self.models.values()
                ],
            }

        @app.post("/v1/embeddings")
        async def _embeddings(request: Request) -> Response:
            return await self._dispatch_json(request, "/v1/embeddings")

        @app.post("/v1/chat/completions")
        async def _chat_completions(request: Request) -> Response:
            return await self._dispatch_json(request, "/v1/chat/completions")

        @app.post("/v1/completions")
        async def _completions(request: Request) -> Response:
            return await self._dispatch_json(request, "/v1/completions")

        @app.post("/v1/audio/transcriptions")
        async def _audio_transcriptions(request: Request) -> Response:
            return await self._dispatch_multipart(request, "/v1/audio/transcriptions")

        @app.post("/v1/audio/speaker-embeddings")
        async def _audio_speaker_embeddings(request: Request) -> Response:
            return await self._dispatch_multipart(request, "/v1/audio/speaker-embeddings")

        @app.post("/v1/audio/gender")
        async def _audio_gender(request: Request) -> Response:
            return await self._dispatch_multipart(request, "/v1/audio/gender")

        @app.post("/v1/audio/diarization")
        async def _audio_diarization(request: Request) -> Response:
            return await self._dispatch_multipart(request, "/v1/audio/diarization")

        return app

    def _status_payload(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "free_vram_mb": self.vram.free_mb(),
            "models": {
                name: {
                    "running": state.running,
                    "active_requests": state.active_requests,
                    "priority": state.spec.priority_name,
                    "port": state.spec.port,
                }
                for name, state in self.models.items()
            },
        }

    def _resolve_state(self, requested_model: str | None, endpoint: str) -> ModelState:
        canonical = self.aliases.get(self._normalize_model(requested_model))
        if canonical and endpoint in self.models[canonical].spec.endpoints:
            return self.models[canonical]

        candidates = [state for state in self.models.values() if endpoint in state.spec.endpoints]
        if requested_model:
            known = ", ".join(sorted(self.aliases))
            raise HTTPException(
                status_code=404,
                detail=f"model '{requested_model}' is not configured for {endpoint}; known models: {known}",
            )
        if len(candidates) == 1:
            return candidates[0]
        raise HTTPException(status_code=400, detail=f"request body must include a model for {endpoint}")

    async def _dispatch_json(self, request: Request, endpoint: str) -> Response:
        raw_body = await request.body()
        try:
            payload = json.loads(raw_body.decode("utf-8") or "{}")
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"invalid JSON body: {exc}") from exc

        state = self._resolve_state(payload.get("model"), endpoint)
        payload["model"] = state.spec.name
        body = json.dumps(payload).encode("utf-8")
        stream = bool(payload.get("stream"))
        return await self._with_model(
            state,
            lambda: self._proxy_raw(request, endpoint, body, state, stream=stream),
        )

    async def _dispatch_multipart(self, request: Request, endpoint: str) -> Response:
        form = await request.form()
        requested_model = str(form.get("model") or "")
        state = self._resolve_state(requested_model, endpoint)

        data: list[tuple[str, str]] = []
        files: list[tuple[str, tuple[str, bytes, str]]] = []
        for key, value in form.multi_items():
            if hasattr(value, "filename"):
                content = await value.read()
                files.append(
                    (
                        key,
                        (
                            value.filename or "upload.bin",
                            content,
                            value.content_type or "application/octet-stream",
                        ),
                    )
                )
            elif key == "model":
                data.append((key, state.spec.name))
            else:
                data.append((key, str(value)))

        if endpoint == "/v1/audio/transcriptions" and not any(is_real_audio_upload(item[1]) for item in files):
            return JSONResponse({"text": "", "model": state.spec.name})

        if not any(key == "model" for key, _ in data):
            data.append(("model", state.spec.name))

        return await self._with_model(
            state,
            lambda: self._proxy_multipart(request, endpoint, data, files, state),
        )

    async def _with_model(self, state: ModelState, call_backend: Any) -> Response:
        state.active_requests += 1
        try:
            await self._ensure_running(state)
            return await call_backend()
        finally:
            state.active_requests -= 1
            state.last_used_monotonic = time.monotonic()

    async def _ensure_running(self, state: ModelState) -> None:
        async with state.lock:
            if state.running:
                return
            await self._make_room_for(state)
            await self._start_model(state)

    async def _make_room_for(self, target: ModelState) -> None:
        if target.spec.estimated_vram_mb <= 0:
            return
        if self._has_room_for(target):
            return

        victims = [
            state
            for state in self.models.values()
            if state.spec.name != target.spec.name and state.running and state.active_requests == 0
        ]
        victims.sort(key=lambda item: (item.spec.priority, item.last_used_monotonic))

        for victim in victims:
            if self._has_room_for(target):
                return
            await self._stop_model(victim, f"freeing VRAM for {target.spec.name}")

        if not self._has_room_for(target):
            free_mb = self.vram.free_mb()
            LOGGER.warning(
                "Starting %s even though free VRAM looks low: free=%s MB, wanted=%s MB + guard=%s MB",
                target.spec.name,
                free_mb,
                target.spec.estimated_vram_mb,
                self.vram_guard_mb,
            )

    def _has_room_for(self, target: ModelState) -> bool:
        free_mb = self.vram.free_mb()
        if free_mb is None:
            return True
        return free_mb >= target.spec.estimated_vram_mb + self.vram_guard_mb

    async def _start_model(self, state: ModelState) -> None:
        spec = state.spec
        state.last_used_monotonic = time.monotonic()
        env = os.environ.copy()
        env.update(
            {
                "MODEL_NAME": spec.name,
                "HOST": spec.host,
                "PORT": str(spec.port),
            }
        )
        cmd = string.Template(spec.cmd).safe_substitute(env)
        LOGGER.info("Starting %s on %s:%s: %s", spec.name, spec.host, spec.port, cmd)

        state.process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
        asyncio.create_task(self._pipe_logs(state))

        try:
            await self._wait_until_ready(state)
            state.last_used_monotonic = time.monotonic()
        except Exception:
            await self._terminate_process(state, "startup failed")
            raise

    async def _pipe_logs(self, state: ModelState) -> None:
        process = state.process
        if process is None or process.stdout is None:
            return
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            LOGGER.info("[%s] %s", state.spec.name, line.decode("utf-8", errors="replace").rstrip())

    async def _wait_until_ready(self, state: ModelState) -> None:
        spec = state.spec
        deadline = time.monotonic() + spec.startup_timeout_sec
        url = f"http://{spec.host}:{spec.port}{spec.health_path}"
        async with httpx.AsyncClient(timeout=5) as client:
            while time.monotonic() < deadline:
                if state.process and state.process.returncode is not None:
                    raise RuntimeError(f"{spec.name} exited during startup with code {state.process.returncode}")
                try:
                    response = await client.get(url)
                    if response.status_code < 400:
                        LOGGER.info("%s is ready", spec.name)
                        return
                except Exception:
                    pass
                await asyncio.sleep(2)
        raise RuntimeError(f"{spec.name} did not become healthy within {spec.startup_timeout_sec}s")

    async def _stop_model(self, state: ModelState, reason: str) -> None:
        async with state.lock:
            await self._terminate_process(state, reason)

    async def _terminate_process(self, state: ModelState, reason: str) -> None:
        process = state.process
        if process is None:
            return
        if process.returncode is not None:
            state.process = None
            return

        LOGGER.info("Stopping %s: %s", state.spec.name, reason)
        try:
            if hasattr(os, "getpgid"):
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            else:
                process.terminate()
            await asyncio.wait_for(process.wait(), timeout=20)
        except asyncio.TimeoutError:
            LOGGER.warning("%s did not stop in time, killing", state.spec.name)
            if hasattr(os, "getpgid"):
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            else:
                process.kill()
            await process.wait()
        finally:
            state.process = None

    async def _proxy_raw(
        self,
        request: Request,
        endpoint: str,
        body: bytes,
        state: ModelState,
        *,
        stream: bool,
    ) -> Response:
        url = self._backend_url(state, endpoint, request.url.query)
        headers = self._proxy_headers(request, content_type="application/json")
        timeout = httpx.Timeout(state.spec.request_timeout_sec)

        if stream:
            client = httpx.AsyncClient(timeout=timeout)
            stream_cm = client.stream(request.method, url, headers=headers, content=body)
            backend_response = await stream_cm.__aenter__()

            async def _iter_response() -> Any:
                try:
                    async for chunk in backend_response.aiter_raw():
                        yield chunk
                finally:
                    await stream_cm.__aexit__(None, None, None)
                    await client.aclose()

            return StreamingResponse(
                _iter_response(),
                status_code=backend_response.status_code,
                headers=self._response_headers(backend_response),
            )

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(request.method, url, headers=headers, content=body)
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=self._response_headers(response),
        )

    async def _proxy_multipart(
        self,
        request: Request,
        endpoint: str,
        data: list[tuple[str, str]],
        files: list[tuple[str, tuple[str, bytes, str]]],
        state: ModelState,
    ) -> Response:
        url = self._backend_url(state, endpoint, request.url.query)
        headers = self._proxy_headers(request)
        timeout = httpx.Timeout(state.spec.request_timeout_sec)
        response = await asyncio.to_thread(
            self._post_multipart_sync,
            url,
            headers,
            data,
            files,
            timeout,
        )
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=self._response_headers(response),
        )

    @staticmethod
    def _post_multipart_sync(
        url: str,
        headers: dict[str, str],
        data: list[tuple[str, str]],
        files: list[tuple[str, tuple[str, bytes, str]]],
        timeout: httpx.Timeout,
    ) -> httpx.Response:
        form_data = {key: value for key, value in data}
        with httpx.Client(timeout=timeout) as client:
            return client.post(url, headers=headers, data=form_data, files=files)

    @staticmethod
    def _proxy_headers(request: Request, content_type: str | None = None) -> dict[str, str]:
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in HEADER_DENYLIST and not key.lower().startswith("x-forwarded-")
        }
        if content_type is not None:
            headers["content-type"] = content_type
        return headers

    @staticmethod
    def _response_headers(response: httpx.Response) -> dict[str, str]:
        headers = {}
        content_type = response.headers.get("content-type")
        if content_type:
            headers["content-type"] = content_type
        return headers

    def _backend_url(self, state: ModelState, endpoint: str, query: str) -> str:
        url = f"http://{state.spec.host}:{state.spec.port}{endpoint}"
        if query:
            url = f"{url}?{query}"
        return url

    async def _idle_reaper(self) -> None:
        while True:
            await asyncio.sleep(self.idle_check_interval)
            now = time.monotonic()
            for state in self.models.values():
                timeout = state.spec.idle_timeout_sec
                if timeout <= 0 or not state.running or state.active_requests > 0:
                    continue
                if now - state.last_used_monotonic >= timeout:
                    await self._stop_model(state, f"idle for {timeout}s")

    async def _prewarm_loop(self) -> None:
        await asyncio.sleep(2)
        while True:
            for state in self.models.values():
                if state.spec.pre_warm and not state.running and state.active_requests == 0:
                    try:
                        await self._ensure_running(state)
                    except Exception as exc:
                        LOGGER.warning("Could not pre-warm %s: %s", state.spec.name, exc)
            await asyncio.sleep(self.prewarm_check_interval)

    def run(self) -> None:
        uvicorn.run(self.app, host=self.host, port=self.port)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/app/config.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    SwapManager(args.config).run()


def is_real_audio_upload(file_tuple: tuple[str, bytes, str]) -> bool:
    filename, content, content_type = file_tuple
    stripped = content.strip()
    if not stripped:
        return False
    if stripped in {b"{}", b"null", b'""'}:
        return False
    if not filename and content_type == "application/octet-stream" and len(stripped) < 16:
        return False
    return True


if __name__ == "__main__":
    main()
