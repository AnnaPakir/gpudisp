from __future__ import annotations

import argparse
import asyncio
import logging
import os
import subprocess
import tempfile
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response


LOGGER = logging.getLogger("gpudisp.gigaam")


def patch_numpy_compat() -> None:
    try:
        import numpy as np
    except Exception:
        return
    if not hasattr(np, "NaN"):
        np.NaN = np.nan


def patch_torch_load_for_gigaam() -> None:
    if getattr(torch.load, "_gpudisp_gigaam_patch", False):
        return

    original_torch_load = torch.load

    def _torch_load_with_pickle(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    _torch_load_with_pickle._gpudisp_gigaam_patch = True  # type: ignore[attr-defined]
    torch.load = _torch_load_with_pickle  # type: ignore[assignment]

    try:
        from torch.serialization import add_safe_globals
        from torch.torch_version import TorchVersion

        add_safe_globals([TorchVersion])
    except Exception as exc:
        LOGGER.warning("Could not add TorchVersion to torch safe globals: %s", exc)


def patch_torchaudio_compat() -> None:
    try:
        import torchaudio
    except Exception as exc:
        LOGGER.warning("Could not import torchaudio for compatibility patch: %s", exc)
        return

    if hasattr(torchaudio, "AudioMetaData"):
        patched_metadata = False
    else:
        try:
            from torchaudio._backend.common import AudioMetaData
        except Exception:
            try:
                from torchaudio.backend.common import AudioMetaData  # type: ignore[no-redef]
            except Exception:
                class AudioMetaData:  # type: ignore[no-redef]
                    pass

        torchaudio.AudioMetaData = AudioMetaData  # type: ignore[attr-defined]
        patched_metadata = True

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = _make_list_audio_backends(torchaudio)  # type: ignore[attr-defined]
        LOGGER.info("Patched torchaudio.list_audio_backends compatibility alias")

    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: None  # type: ignore[attr-defined]

    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda backend=None: None  # type: ignore[attr-defined]

    if hasattr(torchaudio, "info"):
        torchaudio.info = _wrap_torchaudio_backend_kwarg(torchaudio.info)  # type: ignore[assignment]

    if hasattr(torchaudio, "load"):
        torchaudio.load = _wrap_torchaudio_backend_kwarg(torchaudio.load)  # type: ignore[assignment]

    if patched_metadata:
        LOGGER.info("Patched torchaudio.AudioMetaData compatibility alias")


def _make_list_audio_backends(torchaudio_module: Any) -> Any:
    def _list_audio_backends() -> list[str]:
        backend_module = getattr(torchaudio_module, "_backend", None)
        backend_list = getattr(backend_module, "list_audio_backends", None)
        if callable(backend_list):
            try:
                return list(backend_list())
            except Exception:
                pass

        try:
            from torchaudio._backend import utils

            return list(utils.get_available_backends().keys())
        except Exception:
            return ["ffmpeg"]

    return _list_audio_backends


def _wrap_torchaudio_backend_kwarg(func: Any) -> Any:
    if getattr(func, "_gpudisp_backend_kwarg_patch", False):
        return func

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        if "backend" not in kwargs:
            return func(*args, **kwargs)
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            message = str(exc).lower()
            if "backend" not in message:
                raise
            fallback_kwargs = dict(kwargs)
            fallback_kwargs.pop("backend", None)
            return func(*args, **fallback_kwargs)

    _wrapped._gpudisp_backend_kwarg_patch = True  # type: ignore[attr-defined]
    return _wrapped


def patch_huggingface_local_snapshot_download() -> None:
    try:
        import huggingface_hub
    except Exception as exc:
        LOGGER.warning("Could not import huggingface_hub for compatibility patch: %s", exc)
        return

    current_download = huggingface_hub.hf_hub_download
    if getattr(current_download, "_gpudisp_local_snapshot_patch", False):
        return

    def _hf_hub_download(repo_id: Any, filename: str | None = None, *args: Any, **kwargs: Any) -> str:
        if isinstance(repo_id, str) and os.path.isdir(repo_id):
            if filename:
                local_file = os.path.join(repo_id, filename)
                if os.path.exists(local_file):
                    return local_file
            return repo_id
        return current_download(repo_id, filename, *args, **kwargs)

    _hf_hub_download._gpudisp_local_snapshot_patch = True  # type: ignore[attr-defined]
    huggingface_hub.hf_hub_download = _hf_hub_download  # type: ignore[assignment]
    LOGGER.info("Patched huggingface_hub.hf_hub_download for local snapshot paths")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9102)
    parser.add_argument("--model-name", default="v3_e2e_rnnt")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def make_app(args: argparse.Namespace) -> FastAPI:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    patch_numpy_compat()
    patch_torch_load_for_gigaam()
    patch_torchaudio_compat()
    patch_huggingface_local_snapshot_download()

    try:
        import gigaam
    except Exception as exc:
        raise RuntimeError("Python package 'gigaam' is required for ASR") from exc

    LOGGER.info(
        "GigaAM caches: HF_HOME=%s TORCH_HOME=%s XDG_CACHE_HOME=%s torch_hub=%s",
        os.environ.get("HF_HOME"),
        os.environ.get("TORCH_HOME"),
        os.environ.get("XDG_CACHE_HOME"),
        torch.hub.get_dir(),
    )
    LOGGER.info("Loading GigaAM %s on %s", args.model_name, device)
    model = gigaam.load_model(args.model_name)
    if hasattr(model, "to"):
        moved = model.to(device)
        if moved is not None:
            model = moved
    if hasattr(model, "eval"):
        model.eval()

    app = FastAPI(title="GigaAM transcription", version="0.1.0")

    @app.get("/")
    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "model": args.model_name}

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(
        file: UploadFile = File(...),
        model_name: str = Form(default="gigaam", alias="model"),
        response_format: str = Form(default="json"),
        language: str | None = Form(default=None),
        prompt: str | None = Form(default=None),
        temperature: float | None = Form(default=None),
    ) -> Response:
        suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
        fd, path = tempfile.mkstemp(prefix="gpudisp-audio-", suffix=suffix)
        os.close(fd)
        processed_path: str | None = None
        try:
            with open(path, "wb") as fh:
                fh.write(await file.read())
            processed_path = await asyncio.to_thread(preprocess_audio, path)
            result = await asyncio.to_thread(transcribe, model, processed_path)
            text = extract_text(result)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"transcription failed: {exc}") from exc
        finally:
            if processed_path:
                try:
                    os.remove(processed_path)
                except OSError:
                    pass
            try:
                os.remove(path)
            except OSError:
                pass

        if response_format == "text":
            return PlainTextResponse(text)
        if response_format == "verbose_json":
            return JSONResponse({"text": text, "model": model_name, "raw": serialize_result(result)})
        return JSONResponse({"text": text})

    return app


def preprocess_audio(input_path: str) -> str:
    fd, output_path = tempfile.mkstemp(prefix="gpudisp-audio-normalized-", suffix=".wav")
    os.close(fd)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        input_path,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        "-y",
        output_path,
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        try:
            os.remove(output_path)
        except OSError:
            pass
        detail = exc.stderr.strip() or str(exc)
        raise RuntimeError(f"ffmpeg conversion failed: {detail}") from exc
    return output_path


def transcribe(model: Any, path: str) -> Any:
    mode = os.environ.get("GIGAAM_TRANSCRIBE_MODE", "longform").strip().lower()
    if mode in {"longform", "auto"} and hasattr(model, "transcribe_longform"):
        return model.transcribe_longform(path)
    if hasattr(model, "transcribe"):
        try:
            return model.transcribe(path)
        except Exception as exc:
            if mode == "auto" and hasattr(model, "transcribe_longform") and "transcribe_longform" in str(exc):
                return model.transcribe_longform(path)
            raise
    if hasattr(model, "transcribe_longform"):
        return model.transcribe_longform(path)
    raise RuntimeError("loaded GigaAM model does not expose transcribe(path)")


def extract_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        if "text" in result:
            return str(result["text"])
        if "transcription" in result:
            return str(result["transcription"])
        if "segments" in result:
            return " ".join(extract_text(segment) for segment in result["segments"]).strip()
    if isinstance(result, list):
        parts: list[str] = []
        for item in result:
            parts.append(extract_text(item))
        return " ".join(parts).strip()
    if hasattr(result, "text"):
        return str(result.text)
    if hasattr(result, "transcription"):
        return str(result.transcription)
    return str(result)


def serialize_result(result: Any) -> Any:
    if isinstance(result, (str, int, float, bool)) or result is None:
        return result
    if isinstance(result, dict):
        return {key: serialize_result(value) for key, value in result.items()}
    if isinstance(result, list):
        return [serialize_result(item) for item in result]
    known_fields: dict[str, Any] = {}
    for attr in ("text", "transcription", "start", "end", "boundaries"):
        if hasattr(result, attr):
            try:
                known_fields[attr] = serialize_result(getattr(result, attr))
            except Exception:
                pass
    if hasattr(result, "__dict__"):
        payload = {key: serialize_result(value) for key, value in result.__dict__.items()}
        payload.update({key: value for key, value in known_fields.items() if key not in payload})
        return payload
    if known_fields:
        return known_fields
    return str(result)


def main() -> None:
    args = parse_args()
    app = make_app(args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
