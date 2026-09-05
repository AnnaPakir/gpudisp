# gpudisp

GPU-backed OpenAI-compatible API stack for local model serving, request routing,
and on-demand model swapping.

The project combines LiteLLM, llama.cpp, FastAPI proxy services, and Python model
runners for:

- text embeddings
- image embeddings
- vision-language chat and completions
- speech transcription
- speaker embeddings
- gender classification
- speaker diarization

## Requirements

- Docker and Docker Compose
- NVIDIA GPU drivers
- NVIDIA Container Toolkit
- Traefik, if exposing the public gateway through the included labels

Some audio analysis features require a Hugging Face token with accepted model
terms. Configure it with `HF_TOKEN` in your local `.env` file.

## Configuration

Copy the example environment file and update values for your deployment:

```bash
cp .env.example .env
```

Important settings:

- `GPUDISP_DOMAIN`: public hostname for the gateway
- `LITELLM_MASTER_KEY`: API key required by the public gateway
- `HF_TOKEN`: optional token for gated Hugging Face models
- `DIARIZATION_BACKEND`: `simple` or `pyannote`

Model routing lives in `swap_config.yaml`. LiteLLM model aliases live in
`litellm_config.yaml`.

## Model Files

Place GGUF model files in `models/`, or use:

```bash
bash scripts/download_models.sh
```

See `models/README.md` for the expected files and model cache behavior.

## Run

```bash
docker compose up -d --build
```

Check the stack:

```bash
docker compose ps
bash scripts/check_api.sh path/to/audio.wav
```

## Public Gateway

The public gateway exposes OpenAI-compatible routes through LiteLLM and selected
audio analysis routes through the swap manager. Requests must include:

```text
Authorization: Bearer $LITELLM_MASTER_KEY
```

More deployment notes are available in `docs/public-gateway.ru.md`.
