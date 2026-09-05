#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models}"

mkdir -p "${MODEL_DIR}"

download_file() {
  local url="$1"
  local output="$2"
  local target="${MODEL_DIR}/${output}"
  local partial="${target}.part"

  if [[ -s "${target}" ]]; then
    echo "OK: ${output} already exists"
    return
  fi

  echo
  echo "Downloading ${output}"
  echo "from ${url}"

  if [[ -f "${partial}" ]]; then
    curl -L --fail --continue-at - --output "${partial}" "${url}"
  else
    curl -L --fail --output "${partial}" "${url}"
  fi

  mv "${partial}" "${target}"
  echo "Saved: ${target}"
}

download_file \
  "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf?download=true" \
  "Qwen3-Embedding-0.6B-Q8_0.gguf"

download_file \
  "https://huggingface.co/ggml-org/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf?download=true" \
  "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"

download_file \
  "https://huggingface.co/ggml-org/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf?download=true" \
  "mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf"

download_file \
  "https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/resolve/main/Qwen3VL-8B-Instruct-Q4_K_M.gguf?download=true" \
  "Qwen3VL-8B-Instruct-Q4_K_M.gguf"

download_file \
  "https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf?download=true" \
  "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf"

echo
echo "Downloaded model files:"
ls -lh "${MODEL_DIR}"/*.gguf
