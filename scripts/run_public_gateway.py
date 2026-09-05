from __future__ import annotations

import argparse
import logging
import os
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse


LOGGER = logging.getLogger("gpudisp.public_gateway")
HEADER_DENYLIST = {"host", "content-length"}
RESPONSE_HEADER_ALLOWLIST = {
    "cache-control",
    "content-disposition",
    "content-type",
    "x-request-id",
}
SWAP_PUBLIC_ENDPOINTS = {
    "/v1/audio/speaker-embeddings",
    "/v1/audio/gender",
    "/v1/audio/diarization",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4010)
    parser.add_argument("--litellm-base-url", default=os.environ.get("LITELLM_BASE_URL", "http://litellm:4000"))
    parser.add_argument("--swap-base-url", default=os.environ.get("SWAP_MANAGER_BASE_URL", "http://swap_manager:9000"))
    parser.add_argument("--timeout-sec", type=float, default=float(os.environ.get("PUBLIC_GATEWAY_TIMEOUT_SEC", "1200")))
    return parser.parse_args()


class PublicGateway:
    def __init__(self, args: argparse.Namespace) -> None:
        self.host = args.host
        self.port = args.port
        self.litellm_base_url = str(args.litellm_base_url).rstrip("/")
        self.swap_base_url = str(args.swap_base_url).rstrip("/")
        self.timeout = httpx.Timeout(float(args.timeout_sec), connect=30.0)
        self.master_key = os.environ.get("LITELLM_MASTER_KEY") or ""
        self.app = self._make_app()

    def _make_app(self) -> FastAPI:
        app = FastAPI(
            title="gpudisp public gateway",
            version="0.1.0",
            docs_url=None,
            redoc_url=None,
            openapi_url=None,
        )

        @app.get("/internal/health")
        async def internal_health() -> dict[str, Any]:
            return {"status": "ok", "service": "gpudisp-public-gateway"}

        @app.get("/")
        @app.get("/health")
        async def health(request: Request) -> JSONResponse:
            self._require_auth(request)
            litellm = await self._backend_probe(self.litellm_base_url, path="/v1/models", include_auth=True)
            swap_manager = await self._backend_probe(self.swap_base_url, path="/health", include_auth=False)
            status = "ok" if litellm.get("reachable") and swap_manager.get("reachable") else "degraded"
            return JSONResponse(
                {
                    "status": status,
                    "service": "gpudisp-public-gateway",
                    "routes": {
                        "openai_compatible": self.litellm_base_url,
                        "audio_analysis": self.swap_base_url,
                    },
                    "litellm": litellm,
                    "swap_manager": swap_manager,
                }
            )

        @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
        async def proxy(path: str, request: Request) -> Response:
            if request.method == "OPTIONS":
                return Response(status_code=204)
            self._require_auth(request)
            endpoint = "/" + path
            base_url = self.swap_base_url if endpoint in SWAP_PUBLIC_ENDPOINTS else self.litellm_base_url
            include_auth = base_url == self.litellm_base_url
            return await self._proxy_request(request, base_url, endpoint, include_auth=include_auth)

        return app

    def _require_auth(self, request: Request) -> None:
        if not self.master_key:
            raise HTTPException(status_code=503, detail="LITELLM_MASTER_KEY is not configured")
        auth_header = request.headers.get("authorization", "")
        api_key_header = request.headers.get("x-api-key", "")
        if auth_header == f"Bearer {self.master_key}" or api_key_header == self.master_key:
            return
        raise HTTPException(status_code=401, detail="gpudisp api key is required")

    async def _backend_probe(self, base_url: str, *, path: str, include_auth: bool) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.master_key}"} if include_auth and self.master_key else {}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=2.0)) as client:
                response = await client.get(f"{base_url}{path}", headers=headers)
        except Exception as exc:  # noqa: BLE001
            return {"reachable": False, "path": path, "detail": str(exc)}

        payload: Any
        try:
            payload = response.json()
        except ValueError:
            payload = response.text[:500]
        return {
            "reachable": 200 <= response.status_code < 300,
            "path": path,
            "status_code": response.status_code,
            "payload": payload,
        }

    async def _proxy_request(
        self,
        request: Request,
        base_url: str,
        endpoint: str,
        *,
        include_auth: bool,
    ) -> Response:
        body = await request.body()
        query = request.url.query
        url = f"{base_url}{endpoint}"
        if query:
            url = f"{url}?{query}"
        headers = self._proxy_headers(request, include_auth=include_auth)

        client = httpx.AsyncClient(timeout=self.timeout)
        try:
            backend_request = client.build_request(request.method, url, headers=headers, content=body)
            backend_response = await client.send(backend_request, stream=True)
        except Exception:
            await client.aclose()
            raise

        async def _iter_response() -> Any:
            try:
                async for chunk in backend_response.aiter_raw():
                    yield chunk
            finally:
                await backend_response.aclose()
                await client.aclose()

        return StreamingResponse(
            _iter_response(),
            status_code=backend_response.status_code,
            headers=self._response_headers(backend_response),
        )

    def _proxy_headers(self, request: Request, *, include_auth: bool) -> dict[str, str]:
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in HEADER_DENYLIST and not key.lower().startswith("x-forwarded-")
        }
        if include_auth:
            headers["authorization"] = f"Bearer {self.master_key}"
        else:
            headers.pop("authorization", None)
        return headers

    @staticmethod
    def _response_headers(response: httpx.Response) -> dict[str, str]:
        return {
            key: value
            for key, value in response.headers.items()
            if key.lower() in RESPONSE_HEADER_ALLOWLIST
        }

    def run(self) -> None:
        uvicorn.run(self.app, host=self.host, port=self.port)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    PublicGateway(parse_args()).run()


if __name__ == "__main__":
    main()
