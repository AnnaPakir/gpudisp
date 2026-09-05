from __future__ import annotations

import argparse
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8005)
    parser.add_argument("--target", default="http://embeddings_text:8001")
    parser.add_argument("--timeout", type=int, default=180)
    return parser.parse_args()


def make_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="Embedding request sanitizer", version="0.1.0")

    @app.get("/")
    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "target": args.target}

    @app.post("/v1/embeddings")
    async def embeddings(request: Request) -> Response:
        payload = await request.json()
        sanitized = sanitize_embedding_payload(payload)
        async with httpx.AsyncClient(timeout=args.timeout) as client:
            response = await client.post(
                f"{args.target.rstrip('/')}/v1/embeddings",
                json=sanitized,
                headers=proxy_headers(request),
            )
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "application/json"),
        )

    return app


def sanitize_embedding_payload(payload: dict[str, Any]) -> dict[str, Any]:
    clean = {key: value for key, value in payload.items() if value is not None}
    clean.pop("extra_body", None)
    return clean


def proxy_headers(request: Request) -> dict[str, str]:
    return {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in {"host", "content-length"}
    }


def main() -> None:
    args = parse_args()
    uvicorn.run(make_app(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
