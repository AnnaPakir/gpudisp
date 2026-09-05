# Model files

Put local GGUF files here before starting the stack.

Expected files used by the default configs:

- `Qwen3-Embedding-0.6B-Q8_0.gguf`
- `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf`
- `mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf`
- `Qwen3VL-8B-Instruct-Q4_K_M.gguf`
- `mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf`

The Jina CLIP v2, GigaAM, speaker embedding, and gender Python runners download
their Hugging Face/Python model assets on first start. Their cache is kept under
`models/.cache/` where possible, so later restarts should be faster.

Audio analysis models used by `scripts/run_audio_analysis.py`:

- `speechbrain/spkrec-ecapa-voxceleb`: speaker embeddings, about 0.09 GB on disk.
- `alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech`: gender
  classification, about 2.4 GB in the Hugging Face cache when safetensors are
  downloaded.
- `pyannote/speaker-diarization-community-1`: optional pyannote diarization
  backend. It is downloaded from Hugging Face when `backend=pyannote` is used
  and requires `HF_TOKEN` with accepted model terms.
