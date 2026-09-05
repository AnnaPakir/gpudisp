from __future__ import annotations

import argparse
import base64
import io
import logging
import os
from typing import Any

import numpy as np
import requests
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from transformers import AutoModel


LOGGER = logging.getLogger("gpudisp.jinaclip")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff")


class EmbeddingsRequest(BaseModel):
    model: str | None = None
    input: Any
    encoding_format: str | None = "float"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--model-id", default="jinaai/jina-clip-v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--truncate-dim", type=int, default=512)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--task", default=None)
    return parser.parse_args()


def make_app(args: argparse.Namespace) -> FastAPI:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    dtype = "float16" if device == "cuda" else "float32"

    LOGGER.info("Loading %s on %s", args.model_id, device)
    model = AutoModel.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        dtype=dtype,
    )
    model.to(device)
    model.eval()

    app = FastAPI(title="Jina CLIP v2 embeddings", version="0.1.0")

    @app.get("/")
    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "model": args.model_id}

    @app.post("/v1/embeddings")
    async def embeddings(request: EmbeddingsRequest) -> dict[str, Any]:
        items = request.input if isinstance(request.input, list) else [request.input]
        if not items:
            raise HTTPException(status_code=400, detail="input must not be empty")

        vectors: list[list[float]] = []
        for item in items:
            kind, value = parse_input_item(item)
            try:
                if kind == "image":
                    vector = encode_image(model, value, args)
                else:
                    vector = encode_text(model, str(value), args)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"embedding failed: {exc}") from exc
            vectors.append(vector)

        prompt_tokens = sum(estimate_tokens(item) for item in items)
        return {
            "object": "list",
            "model": request.model or args.model_id,
            "data": [
                {"object": "embedding", "embedding": vector, "index": index}
                for index, vector in enumerate(vectors)
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens,
            },
        }

    return app


def parse_input_item(item: Any) -> tuple[str, Any]:
    if isinstance(item, dict):
        if "text" in item:
            return "text", item["text"]
        if "image" in item:
            return "image", load_image(item["image"])
        if "image_url" in item:
            image_url = item["image_url"]
            if isinstance(image_url, dict):
                image_url = image_url.get("url")
            return "image", load_image(image_url)
        if "url" in item:
            return "image", load_image(item["url"])
        raise HTTPException(status_code=400, detail="dict input must contain text, image, image_url, or url")

    if isinstance(item, str) and looks_like_image(item):
        return "image", load_image(item)
    return "text", item


def looks_like_image(value: str) -> bool:
    lowered = value.lower()
    if lowered.startswith("data:image/"):
        return True
    if lowered.startswith(("http://", "https://")) and lowered.split("?", 1)[0].endswith(IMAGE_EXTENSIONS):
        return True
    return os.path.exists(value)


def load_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail="image input must be a path, URL, data URI, or PIL image")

    if value.startswith("data:image/"):
        try:
            _, encoded = value.split(",", 1)
            raw = base64.b64decode(encoded)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid data URI image") from exc
        return Image.open(io.BytesIO(raw)).convert("RGB")

    if value.startswith(("http://", "https://")):
        response = requests.get(value, timeout=30)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")

    return Image.open(value).convert("RGB")


def encode_text(model: Any, text: str, args: argparse.Namespace) -> list[float]:
    return normalize(to_vector(call_encoder(model.encode_text, [text], args)), args.normalize)


def encode_image(model: Any, image: Image.Image, args: argparse.Namespace) -> list[float]:
    return normalize(to_vector(call_encoder(model.encode_image, [image], args)), args.normalize)


def call_encoder(method: Any, payload: list[Any], args: argparse.Namespace) -> Any:
    attempts = []
    if args.task:
        attempts.append({"task": args.task, "truncate_dim": args.truncate_dim})
    attempts.extend([{"truncate_dim": args.truncate_dim}, {}])
    last_error: Exception | None = None
    with torch.inference_mode():
        for kwargs in attempts:
            try:
                return method(payload, **kwargs)
            except TypeError as exc:
                last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError("encoder did not return a value")


def to_vector(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        array = value.detach().float().cpu().numpy()
    else:
        array = np.asarray(value, dtype=np.float32)
    if array.ndim == 2:
        array = array[0]
    return array.astype(np.float32).tolist()


def normalize(vector: list[float], enabled: bool) -> list[float]:
    if not enabled:
        return vector
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm > 0:
        array = array / norm
    return array.tolist()


def estimate_tokens(item: Any) -> int:
    if isinstance(item, dict):
        text = item.get("text") or item.get("url") or item.get("image") or ""
    else:
        text = str(item)
    return max(1, len(str(text)) // 4)


def main() -> None:
    args = parse_args()
    app = make_app(args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
