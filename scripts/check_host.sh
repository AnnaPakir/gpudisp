#!/usr/bin/env bash
set -euo pipefail

DOMAIN="${1:-gpudisp.example.com}"
TRAEFIK_CONTAINER="${TRAEFIK_CONTAINER:-traefik}"

echo "== GPU =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found"
fi

echo
echo "== Docker =="
docker --version
docker compose version

echo
echo "== Existing Traefik =="
docker ps --filter "name=^/${TRAEFIK_CONTAINER}$" --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"
echo
echo "Traefik networks:"
docker inspect "${TRAEFIK_CONTAINER}" --format '{{range $name, $_ := .NetworkSettings.Networks}}{{$name}}{{"\n"}}{{end}}' || true

echo
echo "== NVIDIA container runtime =="
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi

echo
echo "== DNS =="
getent hosts "${DOMAIN}" || true

echo
echo "== Ports =="
if command -v ss >/dev/null 2>&1; then
  ss -tulpn | grep -E ':(80|443)\s' || true
else
  netstat -tulpn | grep -E ':(80|443)\s' || true
fi

if [[ -f docker-compose.yml ]]; then
  echo
  echo "== Compose services =="
  docker compose ps || true

  echo
  echo "== Public gateway networks =="
  docker inspect gpudisp-public_gateway-1 \
    --format '{{range $name, $_ := .NetworkSettings.Networks}}{{$name}}{{"\n"}}{{end}}' || true

  echo
  echo "== Public gateway Traefik labels =="
  docker inspect gpudisp-public_gateway-1 --format '{{json .Config.Labels}}' || true
fi
