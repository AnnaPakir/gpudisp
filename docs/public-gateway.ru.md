# Публичный GPU API

Ваш публичный домен должен вести в сервис `public_gateway`.

Схема:

```text
internet
  -> Traefik
  -> public_gateway
      -> litellm:4000          стандартные OpenAI-like routes
      -> swap_manager:9000     нестандартные audio-analysis routes
```

Публичные routes через настроенный домен:

- `/v1/embeddings`
- `/v1/chat/completions`
- `/v1/completions`
- `/v1/audio/transcriptions`
- `/v1/audio/speaker-embeddings`
- `/v1/audio/gender`
- `/v1/audio/diarization`

Все публичные запросы используют:

```text
Authorization: Bearer $LITELLM_MASTER_KEY
```

`/health` у `public_gateway` тоже требует этот ключ. Для проверки LiteLLM он дергает
`/v1/models`, а для проверки диспетчера моделей - `swap_manager:9000/health`.

`swap_manager:9000` не публикуется через Traefik напрямую. Он остается внутренним
диспетчером моделей и доступен только контейнерам в сети `gpudisp-internal`.

Проверка после деплоя:

```bash
cd /path/to/gpudisp
set -a
source .env
set +a

docker compose up -d --build
docker compose ps
bash scripts/check_api.sh input.mp3
```
