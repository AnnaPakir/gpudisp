from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch


LOGGER = logging.getLogger("gpudisp.audio_models_check")
DEFAULT_SPEAKER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_GENDER_MODEL = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"


def patch_torch_load() -> None:
    if getattr(torch.load, "_gpudisp_audio_models_patch", False):
        return

    original_torch_load = torch.load

    def _torch_load_with_pickle(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    _torch_load_with_pickle._gpudisp_audio_models_patch = True  # type: ignore[attr-defined]
    torch.load = _torch_load_with_pickle  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default=None, help="Optional audio file path inside the container")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--speaker-model-id", default=os.environ.get("SPEAKER_EMBEDDING_MODEL_ID", DEFAULT_SPEAKER_MODEL))
    parser.add_argument("--gender-model-id", default=os.environ.get("GENDER_MODEL_ID", DEFAULT_GENDER_MODEL))
    parser.add_argument("--speaker-dir", default="/models/speechbrain/spkrec-ecapa-voxceleb")
    parser.add_argument("--sample-sec", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    patch_torch_load()

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    print(json.dumps({"torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "device": device}, ensure_ascii=False))
    print_model_catalog()

    waveform, sample_rate = load_audio_sample(args.audio, args.sample_sec)

    speaker_result = check_speaker_model(args.speaker_model_id, args.speaker_dir, waveform, sample_rate, device, args.download_only)
    gender_result = check_gender_model(args.gender_model_id, waveform, sample_rate, device, args.download_only)

    print(json.dumps({"speaker_model": speaker_result, "gender_model": gender_result}, ensure_ascii=False, indent=2))


def print_model_catalog() -> None:
    catalog = [
        {
            "role": "speaker_embeddings",
            "model": DEFAULT_SPEAKER_MODEL,
            "expected_size": "about 89.1 MB on Hugging Face",
        },
        {
            "role": "gender_classification",
            "model": DEFAULT_GENDER_MODEL,
            "expected_size": "2.53 GB full repository; about 1.26 GB when safetensors weights are used",
        },
    ]
    print(json.dumps({"audio_model_catalog": catalog}, ensure_ascii=False, indent=2))


def check_speaker_model(
    model_id: str,
    savedir: str,
    waveform: torch.Tensor,
    sample_rate: int,
    device: str,
    download_only: bool,
) -> dict[str, Any]:
    from speechbrain.inference import EncoderClassifier

    LOGGER.info("Loading speaker embedding model %s into %s", model_id, savedir)
    classifier = EncoderClassifier.from_hparams(
        source=model_id,
        savedir=savedir,
        run_opts={"device": device},
    )

    result: dict[str, Any] = {
        "id": model_id,
        "savedir": savedir,
        "savedir_size_mb": round(size_mb(Path(savedir)), 2),
    }

    if download_only:
        return result

    audio = ensure_mono_16k(waveform, sample_rate).to(device)
    with torch.inference_mode():
        embedding = classifier.encode_batch(audio)
    result.update(
        {
            "embedding_shape": list(embedding.shape),
            "embedding_dtype": str(embedding.dtype),
            "embedding_device": str(embedding.device),
        }
    )
    return result


def check_gender_model(
    model_id: str,
    waveform: torch.Tensor,
    sample_rate: int,
    device: str,
    download_only: bool,
) -> dict[str, Any]:
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    LOGGER.info("Loading gender model %s", model_id)
    extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModelForAudioClassification.from_pretrained(model_id, use_safetensors=True)
    model.to(device)
    model.eval()

    cache_dir = hf_cache_dir_for(model_id)
    result: dict[str, Any] = {
        "id": model_id,
        "cache_dir": str(cache_dir) if cache_dir else None,
        "cache_size_mb": round(size_mb(cache_dir), 2) if cache_dir else None,
    }

    if download_only:
        return result

    audio = ensure_mono_16k(waveform, sample_rate).squeeze(0).cpu().numpy()
    inputs = extractor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        output = model(**inputs)
        probabilities = torch.softmax(output.logits, dim=-1)[0]
        label_id = int(torch.argmax(probabilities).item())
        score = float(probabilities[label_id].item())
    result.update({"label": model.config.id2label.get(label_id, str(label_id)), "score": round(score, 4)})
    return result


def load_audio_sample(audio_path: str | None, sample_sec: float) -> tuple[torch.Tensor, int]:
    if audio_path:
        return load_audio_with_ffmpeg(Path(audio_path), sample_sec)

    samples = int(16000 * sample_sec)
    waveform = torch.zeros(1, samples, dtype=torch.float32)
    return waveform, 16000


def load_audio_with_ffmpeg(path: Path, sample_sec: float) -> tuple[torch.Tensor, int]:
    if not path.exists():
        raise FileNotFoundError(path)

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-t",
        str(sample_sec),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-f",
        "s16le",
        "-",
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return torch.from_numpy(audio).unsqueeze(0), 16000


def ensure_mono_16k(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    import torchaudio

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    return waveform.contiguous()


def hf_cache_dir_for(model_id: str) -> Path | None:
    hf_home = Path(os.environ.get("HF_HOME", "/models/.cache/huggingface"))
    repo_dir = hf_home / "hub" / f"models--{model_id.replace('/', '--')}"
    return repo_dir if repo_dir.exists() else None


def size_mb(path: Path | None) -> float:
    if path is None or not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_size / 1024 / 1024
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / 1024 / 1024


if __name__ == "__main__":
    main()
