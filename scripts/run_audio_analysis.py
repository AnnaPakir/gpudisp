from __future__ import annotations

import argparse
import logging
import math
import os
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


LOGGER = logging.getLogger("gpudisp.audio_analysis")
DEFAULT_SPEAKER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_GENDER_MODEL = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
DEFAULT_PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"
SAMPLE_RATE = 16000


@dataclass(frozen=True)
class AudioChunk:
    start_sec: float
    end_sec: float
    waveform: torch.Tensor
    rms: float

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


def patch_torch_load() -> None:
    if getattr(torch.load, "_gpudisp_audio_analysis_patch", False):
        return

    original_torch_load = torch.load

    def _torch_load_with_pickle(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    _torch_load_with_pickle._gpudisp_audio_analysis_patch = True  # type: ignore[attr-defined]
    torch.load = _torch_load_with_pickle  # type: ignore[assignment]

    try:
        from torch.serialization import add_safe_globals
        from torch.torch_version import TorchVersion

        add_safe_globals([TorchVersion])
    except Exception as exc:  # pragma: no cover - optional compatibility path
        LOGGER.warning("Could not add TorchVersion to torch safe globals: %s", exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9103)
    parser.add_argument("--speaker-model-id", default=os.environ.get("SPEAKER_EMBEDDING_MODEL_ID", DEFAULT_SPEAKER_MODEL))
    parser.add_argument("--gender-model-id", default=os.environ.get("GENDER_MODEL_ID", DEFAULT_GENDER_MODEL))
    parser.add_argument("--pyannote-model-id", default=os.environ.get("PYANNOTE_MODEL_ID", DEFAULT_PYANNOTE_MODEL))
    parser.add_argument("--diarization-backend", default=os.environ.get("DIARIZATION_BACKEND", "simple"))
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or None)
    parser.add_argument("--speaker-dir", default="/models/speechbrain/spkrec-ecapa-voxceleb")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preload-gender", action="store_true")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    return parser.parse_args()


class AudioAnalysisRuntime:
    def __init__(self, args: argparse.Namespace) -> None:
        patch_torch_load()
        self.args = args
        self.device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
        self.speechbrain_device = "cuda:0" if self.device == "cuda" else self.device
        self.speaker_lock = threading.Lock()
        self.gender_lock = threading.Lock()
        self.pyannote_lock = threading.Lock()
        self.gender_extractor: Any | None = None
        self.gender_model: Any | None = None
        self.pyannote_pipeline: Any | None = None

        LOGGER.info("Loading speaker embedding model %s on %s", args.speaker_model_id, self.speechbrain_device)
        from speechbrain.inference import EncoderClassifier

        self.speaker_classifier = EncoderClassifier.from_hparams(
            source=args.speaker_model_id,
            savedir=args.speaker_dir,
            run_opts={"device": self.speechbrain_device},
        )
        if args.preload_gender:
            self._ensure_gender_model()

    def speaker_embedding(self, audio: np.ndarray, *, normalize: bool) -> list[float]:
        waveform = torch.from_numpy(audio.astype(np.float32, copy=False)).unsqueeze(0).to(self.speechbrain_device)
        with self.speaker_lock, torch.inference_mode():
            embedding = self.speaker_classifier.encode_batch(waveform)
        vector = embedding.detach().float().cpu().reshape(-1).numpy()
        if normalize:
            vector = normalize_vector(vector)
        return vector.astype(np.float32).tolist()

    def gender(self, audio: np.ndarray) -> dict[str, Any]:
        extractor, model = self._ensure_gender_model()
        inputs = extractor(audio.astype(np.float32, copy=False), sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with self.gender_lock, torch.inference_mode():
            output = model(**inputs)
            probabilities = torch.softmax(output.logits, dim=-1)[0].detach().float().cpu().numpy()

        label_id = int(np.argmax(probabilities))
        raw_label = str(model.config.id2label.get(label_id, label_id))
        label = normalize_gender_label(raw_label)
        return {
            "gender": label,
            "label": label,
            "raw_label": raw_label,
            "score": round(float(probabilities[label_id]), 4),
            "probabilities": {
                normalize_gender_label(str(model.config.id2label.get(index, index))): round(float(score), 4)
                for index, score in enumerate(probabilities)
            },
        }

    def diarize_simple(
        self,
        audio: np.ndarray,
        *,
        chunk_sec: float,
        min_chunk_sec: float,
        min_speakers: int,
        max_speakers: int,
        silence_rms_threshold: float,
        min_cluster_score: float,
        merge_gap_sec: float,
    ) -> dict[str, Any]:
        chunks = make_chunks(
            audio,
            chunk_sec=chunk_sec,
            min_chunk_sec=min_chunk_sec,
            silence_rms_threshold=silence_rms_threshold,
        )
        if not chunks:
            return {
                "object": "audio.diarization",
                "model": self.args.speaker_model_id,
                "segments": [],
                "num_speakers": 0,
                "duration_sec": round(len(audio) / SAMPLE_RATE, 3),
                "strategy": "speechbrain-ecapa+agglomerative",
            }

        vectors = self.speaker_embeddings_for_chunks(chunks)
        labels, cluster_info = choose_cluster_labels(
            vectors,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            min_cluster_score=min_cluster_score,
        )
        segments = merge_labeled_chunks(chunks, labels, merge_gap_sec=merge_gap_sec)
        return {
            "object": "audio.diarization",
            "model": self.args.speaker_model_id,
            "segments": segments,
            "num_speakers": len({segment["speaker_id"] for segment in segments}),
            "duration_sec": round(len(audio) / SAMPLE_RATE, 3),
            "chunk_sec": chunk_sec,
            "speech_chunks": len(chunks),
            "strategy": "speechbrain-ecapa+agglomerative",
            "cluster_info": cluster_info,
        }

    def diarize_pyannote(
        self,
        audio: np.ndarray,
        *,
        min_speakers: int,
        max_speakers: int,
        merge_gap_sec: float,
        exclusive: bool,
    ) -> dict[str, Any]:
        pipeline = self._ensure_pyannote_pipeline()
        waveform = torch.from_numpy(audio.astype(np.float32, copy=False)).unsqueeze(0)
        payload = {"waveform": waveform, "sample_rate": SAMPLE_RATE}
        kwargs: dict[str, Any] = {}
        if min_speakers > 0:
            kwargs["min_speakers"] = min_speakers
        if max_speakers > 0:
            kwargs["max_speakers"] = max_speakers

        with self.pyannote_lock, torch.inference_mode():
            output = pipeline(payload, **kwargs)

        annotation = _select_pyannote_annotation(output, exclusive=exclusive)
        raw_segments = _pyannote_annotation_to_segments(annotation)
        segments = merge_pyannote_segments(raw_segments, merge_gap_sec=merge_gap_sec)
        return {
            "object": "audio.diarization",
            "model": self.args.pyannote_model_id,
            "segments": segments,
            "num_speakers": len({segment["speaker_id"] for segment in segments}),
            "duration_sec": round(len(audio) / SAMPLE_RATE, 3),
            "strategy": "pyannote",
            "exclusive": exclusive,
            "backend": "pyannote",
        }

    def speaker_embeddings_for_chunks(self, chunks: list[AudioChunk]) -> np.ndarray:
        batch_size = max(1, int(self.args.embedding_batch_size))
        vectors: list[np.ndarray] = []
        with self.speaker_lock, torch.inference_mode():
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                max_len = max(chunk.waveform.numel() for chunk in batch)
                padded = torch.zeros((len(batch), max_len), dtype=torch.float32)
                lengths = torch.empty((len(batch),), dtype=torch.float32)
                for index, chunk in enumerate(batch):
                    sample_count = chunk.waveform.numel()
                    padded[index, :sample_count] = chunk.waveform
                    lengths[index] = sample_count / max_len
                padded = padded.to(self.speechbrain_device)
                lengths = lengths.to(self.speechbrain_device)
                try:
                    embedding = self.speaker_classifier.encode_batch(padded, wav_lens=lengths)
                except TypeError:
                    embedding = self.speaker_classifier.encode_batch(padded)
                embedding_np = embedding.detach().float().cpu().reshape(len(batch), -1).numpy()
                vectors.extend(normalize_vector(row) for row in embedding_np)
        return np.vstack(vectors).astype(np.float32)

    def _ensure_gender_model(self) -> tuple[Any, Any]:
        if self.gender_extractor is not None and self.gender_model is not None:
            return self.gender_extractor, self.gender_model
        with self.gender_lock:
            if self.gender_extractor is not None and self.gender_model is not None:
                return self.gender_extractor, self.gender_model

            LOGGER.info("Loading gender model %s on %s", self.args.gender_model_id, self.device)
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

            self.gender_extractor = AutoFeatureExtractor.from_pretrained(self.args.gender_model_id)
            self.gender_model = AutoModelForAudioClassification.from_pretrained(
                self.args.gender_model_id,
                use_safetensors=True,
            )
            self.gender_model.to(self.device)
            self.gender_model.eval()
            return self.gender_extractor, self.gender_model

    def _ensure_pyannote_pipeline(self) -> Any:
        if self.pyannote_pipeline is not None:
            return self.pyannote_pipeline
        with self.pyannote_lock:
            if self.pyannote_pipeline is not None:
                return self.pyannote_pipeline
            if not self.args.hf_token:
                raise RuntimeError("HF_TOKEN is required for pyannote diarization")

            LOGGER.info("Loading pyannote diarization model %s on %s", self.args.pyannote_model_id, self.device)
            from pyannote.audio import Pipeline

            try:
                pipeline = Pipeline.from_pretrained(self.args.pyannote_model_id, token=self.args.hf_token)
            except TypeError:
                pipeline = Pipeline.from_pretrained(self.args.pyannote_model_id, use_auth_token=self.args.hf_token)
            if self.device != "cpu":
                pipeline.to(torch.device(self.device))
            self.pyannote_pipeline = pipeline
            return self.pyannote_pipeline


def make_app(args: argparse.Namespace) -> FastAPI:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    runtime = AudioAnalysisRuntime(args)
    app = FastAPI(title="gpudisp audio analysis", version="0.1.0")

    @app.get("/")
    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "device": runtime.device,
            "speaker_model": args.speaker_model_id,
            "gender_model": args.gender_model_id,
            "gender_loaded": runtime.gender_model is not None,
            "diarization_backend": args.diarization_backend,
            "pyannote_model": args.pyannote_model_id,
            "pyannote_loaded": runtime.pyannote_pipeline is not None,
        }

    @app.post("/v1/audio/speaker-embeddings")
    async def speaker_embeddings(
        file: UploadFile = File(...),
        model_name: str = Form(default="audio-analysis", alias="model"),
        normalize: bool = Form(default=False),
    ) -> dict[str, Any]:
        audio = await load_upload_audio(file)
        vector = await run_in_thread(runtime.speaker_embedding, audio, normalize=normalize)
        return {
            "object": "audio.speaker_embedding",
            "model": args.speaker_model_id,
            "requested_model": model_name,
            "embedding": vector,
            "embedding_dim": len(vector),
            "data": [{"object": "speaker_embedding", "embedding": vector, "index": 0}],
        }

    @app.post("/v1/audio/gender")
    async def gender(
        file: UploadFile = File(...),
        model_name: str = Form(default="audio-analysis", alias="model"),
        sample_sec: float = Form(default=8.0),
        offset_sec: float = Form(default=0.0),
    ) -> dict[str, Any]:
        audio = await load_upload_audio(file)
        sample = crop_audio(audio, offset_sec=offset_sec, sample_sec=sample_sec)
        if sample.size == 0:
            raise HTTPException(status_code=400, detail="audio sample is empty")
        result = await run_in_thread(runtime.gender, sample)
        return {
            "object": "audio.gender",
            "model": args.gender_model_id,
            "requested_model": model_name,
            **result,
        }

    @app.post("/v1/audio/diarization")
    async def diarization(
        file: UploadFile = File(...),
        model_name: str = Form(default="audio-analysis", alias="model"),
        chunk_sec: float = Form(default=3.0),
        min_chunk_sec: float = Form(default=1.0),
        min_speakers: int = Form(default=1),
        max_speakers: int = Form(default=8),
        silence_rms_threshold: float = Form(default=0.003),
        min_cluster_score: float = Form(default=0.05),
        merge_gap_sec: float = Form(default=0.25),
        backend: str | None = Form(default=None),
        exclusive: bool = Form(default=True),
        fallback_backend: bool = Form(default=True),
    ) -> dict[str, Any]:
        if chunk_sec <= 0:
            raise HTTPException(status_code=400, detail="chunk_sec must be positive")
        if min_chunk_sec <= 0:
            raise HTTPException(status_code=400, detail="min_chunk_sec must be positive")
        if max_speakers < 1:
            raise HTTPException(status_code=400, detail="max_speakers must be positive")

        audio = await load_upload_audio(file)
        selected_backend = normalize_diarization_backend(backend or args.diarization_backend)
        try:
            if selected_backend == "pyannote":
                result = await run_in_thread(
                    runtime.diarize_pyannote,
                    audio,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    merge_gap_sec=merge_gap_sec,
                    exclusive=exclusive,
                )
            else:
                result = await run_in_thread(
                    runtime.diarize_simple,
                    audio,
                    chunk_sec=chunk_sec,
                    min_chunk_sec=min_chunk_sec,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    silence_rms_threshold=silence_rms_threshold,
                    min_cluster_score=min_cluster_score,
                    merge_gap_sec=merge_gap_sec,
                )
        except Exception as exc:
            if selected_backend != "pyannote" or not fallback_backend:
                raise HTTPException(status_code=500, detail=f"{selected_backend} diarization failed: {exc}") from exc
            LOGGER.exception("Pyannote diarization failed; falling back to simple backend")
            result = await run_in_thread(
                runtime.diarize_simple,
                audio,
                chunk_sec=chunk_sec,
                min_chunk_sec=min_chunk_sec,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                silence_rms_threshold=silence_rms_threshold,
                min_cluster_score=min_cluster_score,
                merge_gap_sec=merge_gap_sec,
            )
            result["fallback_from"] = "pyannote"
            result["fallback_reason"] = str(exc)
        result["requested_model"] = model_name
        result["requested_backend"] = selected_backend
        return result

    return app


async def run_in_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
    import asyncio

    return await asyncio.to_thread(func, *args, **kwargs)


async def load_upload_audio(file: UploadFile) -> np.ndarray:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="uploaded audio file is empty")
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    fd, path = tempfile.mkstemp(prefix="gpudisp-analysis-", suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
        return decode_audio_file(path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not decode audio: {exc}") from exc
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def decode_audio_bytes(raw: bytes) -> np.ndarray:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "1",
        "-f",
        "s16le",
        "pipe:1",
    ]
    result = subprocess.run(command, input=raw, check=True, capture_output=True)
    if not result.stdout:
        raise RuntimeError("ffmpeg returned empty audio")
    audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


def decode_audio_file(path: str) -> np.ndarray:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        path,
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "1",
        "-f",
        "s16le",
        "pipe:1",
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    if not result.stdout:
        raise RuntimeError("ffmpeg returned empty audio")
    audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


def crop_audio(audio: np.ndarray, *, offset_sec: float, sample_sec: float) -> np.ndarray:
    start = max(0, int(offset_sec * SAMPLE_RATE))
    if sample_sec <= 0:
        return audio[start:]
    end = min(audio.size, start + int(sample_sec * SAMPLE_RATE))
    return audio[start:end]


def make_chunks(
    audio: np.ndarray,
    *,
    chunk_sec: float,
    min_chunk_sec: float,
    silence_rms_threshold: float,
) -> list[AudioChunk]:
    chunk_samples = max(1, int(chunk_sec * SAMPLE_RATE))
    min_samples = max(1, int(min_chunk_sec * SAMPLE_RATE))
    chunks: list[AudioChunk] = []
    for start in range(0, audio.size, chunk_samples):
        end = min(audio.size, start + chunk_samples)
        if end - start < min_samples:
            continue
        piece = audio[start:end]
        rms = float(math.sqrt(float(np.mean(np.square(piece)))) if piece.size else 0.0)
        if rms < silence_rms_threshold:
            continue
        chunks.append(
            AudioChunk(
                start_sec=round(start / SAMPLE_RATE, 3),
                end_sec=round(end / SAMPLE_RATE, 3),
                waveform=torch.from_numpy(piece.astype(np.float32, copy=False)),
                rms=rms,
            )
        )
    return chunks


def choose_cluster_labels(
    vectors: np.ndarray,
    *,
    min_speakers: int,
    max_speakers: int,
    min_cluster_score: float,
) -> tuple[list[int], dict[str, Any]]:
    count = len(vectors)
    min_k = max(1, min(min_speakers, count))
    max_k = max(min_k, min(max_speakers, count))
    if count <= 1 or max_k <= 1:
        return [0] * count, {"selected_speakers": 1 if count else 0, "score": None}

    best_labels: list[int] | None = None
    best_score: float | None = None
    best_k = 1
    upper_for_score = min(max_k, count - 1)
    for k in range(max(2, min_k), upper_for_score + 1):
        try:
            labels = cluster_vectors(vectors, k)
            if len(set(labels)) < 2:
                continue
            score = float(silhouette_score(vectors, labels, metric="cosine"))
        except Exception as exc:
            LOGGER.debug("Could not score %s speaker clusters: %s", k, exc)
            continue
        if best_score is None or score > best_score:
            best_labels = labels
            best_score = score
            best_k = k

    if best_labels is None:
        if min_k > 1 and count >= min_k:
            labels = cluster_vectors(vectors, min_k)
            return labels, {"selected_speakers": len(set(labels)), "score": None, "forced_min_speakers": True}
        return [0] * count, {"selected_speakers": 1, "score": None}

    if min_k <= 1 and best_score is not None and best_score < min_cluster_score:
        return [0] * count, {
            "selected_speakers": 1,
            "score": round(best_score, 4),
            "best_scored_speakers": best_k,
            "reason": "below_min_cluster_score",
        }

    return best_labels, {"selected_speakers": best_k, "score": round(best_score, 4) if best_score is not None else None}


def cluster_vectors(vectors: np.ndarray, k: int) -> list[int]:
    try:
        clusterer = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
    except TypeError:  # pragma: no cover - sklearn compatibility path
        clusterer = AgglomerativeClustering(n_clusters=k, affinity="cosine", linkage="average")
    return [int(label) for label in clusterer.fit_predict(vectors)]


def merge_labeled_chunks(chunks: list[AudioChunk], labels: list[int], *, merge_gap_sec: float) -> list[dict[str, Any]]:
    label_names: dict[int, str] = {}
    segments: list[dict[str, Any]] = []
    for chunk, label in zip(chunks, labels):
        if label not in label_names:
            label_names[label] = f"speaker_{len(label_names)}"
        speaker_id = label_names[label]
        if segments and segments[-1]["speaker_id"] == speaker_id and chunk.start_sec <= segments[-1]["end_sec"] + merge_gap_sec:
            segments[-1]["end_sec"] = chunk.end_sec
            segments[-1]["duration_sec"] = round(segments[-1]["end_sec"] - segments[-1]["start_sec"], 3)
            segments[-1]["chunk_count"] += 1
            segments[-1]["rms"] = round(max(float(segments[-1]["rms"]), chunk.rms), 5)
            continue
        segments.append(
            {
                "start_sec": chunk.start_sec,
                "end_sec": chunk.end_sec,
                "duration_sec": round(chunk.duration_sec, 3),
                "speaker_id": speaker_id,
                "score": None,
                "chunk_count": 1,
                "rms": round(chunk.rms, 5),
            }
        )
    return segments


def _select_pyannote_annotation(output: Any, *, exclusive: bool) -> Any:
    if exclusive and hasattr(output, "exclusive_speaker_diarization"):
        annotation = getattr(output, "exclusive_speaker_diarization")
        if annotation is not None:
            return annotation
    if hasattr(output, "speaker_diarization"):
        annotation = getattr(output, "speaker_diarization")
        if annotation is not None:
            return annotation
    return output


def _pyannote_annotation_to_segments(annotation: Any) -> list[dict[str, Any]]:
    label_names: dict[str, str] = {}
    segments: list[dict[str, Any]] = []

    if hasattr(annotation, "itertracks"):
        iterator = annotation.itertracks(yield_label=True)
        for turn, _, speaker in iterator:
            segments.append(_pyannote_segment_row(turn.start, turn.end, str(speaker), label_names))
        return sorted(segments, key=lambda row: (row["start_sec"], row["end_sec"]))

    for item in annotation:
        try:
            turn, speaker = item
        except (TypeError, ValueError):
            continue
        segments.append(_pyannote_segment_row(turn.start, turn.end, str(speaker), label_names))
    return sorted(segments, key=lambda row: (row["start_sec"], row["end_sec"]))


def _pyannote_segment_row(start_sec: float, end_sec: float, speaker: str, label_names: dict[str, str]) -> dict[str, Any]:
    if speaker not in label_names:
        label_names[speaker] = f"speaker_{len(label_names)}"
    start = round(float(start_sec), 3)
    end = round(float(end_sec), 3)
    return {
        "start_sec": start,
        "end_sec": end,
        "duration_sec": round(max(0.0, end - start), 3),
        "speaker_id": label_names[speaker],
        "raw_speaker_id": speaker,
        "score": None,
        "chunk_count": 1,
    }


def merge_pyannote_segments(segments: list[dict[str, Any]], *, merge_gap_sec: float) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for segment in sorted(segments, key=lambda row: (row["start_sec"], row["end_sec"])):
        if segment["duration_sec"] <= 0:
            continue
        if (
            merged
            and merged[-1]["speaker_id"] == segment["speaker_id"]
            and segment["start_sec"] <= merged[-1]["end_sec"] + merge_gap_sec
        ):
            merged[-1]["end_sec"] = max(merged[-1]["end_sec"], segment["end_sec"])
            merged[-1]["duration_sec"] = round(merged[-1]["end_sec"] - merged[-1]["start_sec"], 3)
            merged[-1]["chunk_count"] += 1
            continue
        merged.append(dict(segment))
    return merged


def normalize_diarization_backend(value: str) -> str:
    lowered = (value or "simple").strip().lower()
    if lowered in {"simple", "speechbrain", "ecapa", "speechbrain-ecapa"}:
        return "simple"
    if lowered in {"pyannote", "community-1", "community1"}:
        return "pyannote"
    raise HTTPException(status_code=400, detail="backend must be simple or pyannote")


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        return vector / norm
    return vector


def normalize_gender_label(label: str) -> str:
    lowered = label.strip().lower()
    if lowered in {"m", "male", "man", "masculine"}:
        return "male"
    if lowered in {"f", "female", "woman", "feminine"}:
        return "female"
    return lowered


def main() -> None:
    args = parse_args()
    app = make_app(args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
