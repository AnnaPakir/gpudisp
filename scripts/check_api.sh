#!/usr/bin/env bash
set -euo pipefail

DOMAIN="${GPUDISP_DOMAIN:-gpudisp.example.com}"
BASE_URL="${BASE_URL:-https://${DOMAIN}}"
AUDIO_FILE="${1:-}"
CURL_MAX_TIME="${CURL_MAX_TIME:-900}"
OUT_DIR="${OUT_DIR:-/tmp/gpudisp-api-check}"
TEST_IMAGE_DATA_URI="${TEST_IMAGE_DATA_URI:-data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=}"

if [[ -z "${LITELLM_MASTER_KEY:-}" ]]; then
  echo "ERROR: LITELLM_MASTER_KEY is not set."
  echo "Run first: set -a && source .env && set +a"
  exit 2
fi

mkdir -p "${OUT_DIR}"

FAILURES=0

run_request() {
  local name="$1"
  shift
  local output="${OUT_DIR}/${name}.json"
  local code

  echo
  echo "== ${name} =="
  set +e
  code="$(curl -sS --max-time "${CURL_MAX_TIME}" -w "%{http_code}" -o "${output}" "$@")"
  local curl_exit=$?
  set -e

  if [[ ${curl_exit} -ne 0 ]]; then
    echo "curl failed with exit code ${curl_exit}"
    FAILURES=$((FAILURES + 1))
    return
  fi

  local bytes
  bytes="$(wc -c < "${output}" | tr -d ' ')"
  echo "HTTP ${code}; saved ${bytes} bytes to ${output}"
  head -c 800 "${output}" || true
  echo

  if [[ "${code}" -lt 200 || "${code}" -ge 300 ]]; then
    FAILURES=$((FAILURES + 1))
  fi
}

AUTH_HEADER="Authorization: Bearer ${LITELLM_MASTER_KEY}"
IMAGE_EMBEDDING_PAYLOAD="$(printf '{"model":"image-embedding","input":[{"image_url":{"url":"%s"}}]}' "${TEST_IMAGE_DATA_URI}")"
QWEN_VL_IMAGE_PAYLOAD="$(printf '{"model":"qwen-vl","messages":[{"role":"user","content":[{"type":"text","text":"Describe this image in one short sentence."},{"type":"image_url","image_url":{"url":"%s"}}]}],"max_tokens":80}' "${TEST_IMAGE_DATA_URI}")"
QWEN3_VL_IMAGE_PAYLOAD="$(printf '{"model":"qwen3-vl","messages":[{"role":"user","content":[{"type":"text","text":"Describe this image in one short sentence."},{"type":"image_url","image_url":{"url":"%s"}}]}],"max_tokens":80}' "${TEST_IMAGE_DATA_URI}")"

if command -v docker >/dev/null 2>&1 && [[ -f docker-compose.yml ]]; then
  echo "== docker compose ps =="
  docker compose ps || true
fi

run_request "health" \
  "${BASE_URL}/health" \
  -H "${AUTH_HEADER}"

run_request "models" \
  "${BASE_URL}/v1/models" \
  -H "${AUTH_HEADER}"

run_request "text_embedding" \
  "${BASE_URL}/v1/embeddings" \
  -H "${AUTH_HEADER}" \
  -H "Content-Type: application/json" \
  -d '{"model":"text-embedding","input":"test text"}'

run_request "image_embedding" \
  "${BASE_URL}/v1/embeddings" \
  -H "${AUTH_HEADER}" \
  -H "Content-Type: application/json" \
  -d "${IMAGE_EMBEDDING_PAYLOAD}"

run_request "qwen_vl_text" \
  "${BASE_URL}/v1/chat/completions" \
  -H "${AUTH_HEADER}" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen-vl","messages":[{"role":"user","content":"Answer in one short sentence: are you working?"}],"max_tokens":64}'

run_request "qwen_vl_image" \
  "${BASE_URL}/v1/chat/completions" \
  -H "${AUTH_HEADER}" \
  -H "Content-Type: application/json" \
  -d "${QWEN_VL_IMAGE_PAYLOAD}"

run_request "qwen3_vl_text" \
  "${BASE_URL}/v1/chat/completions" \
  -H "${AUTH_HEADER}" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-vl","messages":[{"role":"user","content":"Answer in one short sentence: are you working?"}],"max_tokens":64}'

run_request "qwen3_vl_image" \
  "${BASE_URL}/v1/chat/completions" \
  -H "${AUTH_HEADER}" \
  -H "Content-Type: application/json" \
  -d "${QWEN3_VL_IMAGE_PAYLOAD}"

if [[ -n "${AUDIO_FILE}" ]]; then
  if [[ -f "${AUDIO_FILE}" ]]; then
    run_request "gigaam" \
      "${BASE_URL}/v1/audio/transcriptions" \
      -H "${AUTH_HEADER}" \
      -F model=gigaam \
      -F "file=@${AUDIO_FILE}"

    if [[ "${CHECK_AUDIO_ANALYSIS:-true}" == "true" ]]; then
      run_request "speaker_embeddings" \
        "${BASE_URL}/v1/audio/speaker-embeddings" \
        -H "${AUTH_HEADER}" \
        -F "file=@${AUDIO_FILE}"

      run_request "gender" \
        "${BASE_URL}/v1/audio/gender" \
        -H "${AUTH_HEADER}" \
        -F "file=@${AUDIO_FILE}"

      run_request "diarization" \
        "${BASE_URL}/v1/audio/diarization" \
        -H "${AUTH_HEADER}" \
        -F "file=@${AUDIO_FILE}"
    else
      echo
      echo "== audio-analysis =="
      echo "Skipped: CHECK_AUDIO_ANALYSIS is not true."
    fi
  else
    echo
    echo "== gigaam =="
    echo "Skipped: audio file not found: ${AUDIO_FILE}"
    FAILURES=$((FAILURES + 1))
  fi
else
  echo
  echo "== gigaam =="
  echo "Skipped: pass an audio file path as the first argument to test transcription."
fi

echo
if [[ ${FAILURES} -eq 0 ]]; then
  echo "All requested checks passed."
else
  echo "${FAILURES} check(s) failed. See saved responses in ${OUT_DIR}."
  exit 1
fi
