# Inference Server

A small distributed inference stack with a FastAPI gateway, a local PyTorch embedding model server, and optional NGINX reverse proxy.

## Overview

This repository includes:

- `gateway/`: FastAPI service that exposes `/v1/embeddings` and forwards requests to the model server.
- `model-server/`: FastAPI embedding server that loads a local transformer model and returns embeddings.
- `nginx/`: Optional reverse proxy configuration for routing traffic to the gateway.
- `docker-compose.yml`: Compose configuration for the three services.
- `.github/workflows/docker-publish.yml`: GitHub Actions workflow to build and push the service images.

## Architecture

- `gateway` handles authentication via `API_KEY` and proxies requests to the model server.
- `model-server` loads a local model from `./models/all-MiniLM-L6-v2` and serves embeddings at `/v1/embeddings`.
- `nginx` is provided as a lightweight reverse proxy in front of the gateway.

## Prerequisites

- Docker Engine and Docker Compose
- Python 3.11 for local development (optional)
- A local copy of the model at `./models/all-MiniLM-L6-v2`
- Docker registry credentials for image publishing

## Local development

1. Create a `.env` file or export variables locally:

```bash
MODEL_PATH=./models/all-MiniLM-L6-v2
MODEL_SERVER_URL=http://model-server:8001
API_KEY=dev-secret-key
```

2. Start the stack with Docker Compose:

```bash
docker compose -f docker-compose.yml up --build
```

3. Verify services:

- Gateway health: `http://localhost:8000/health`
- Model server health: `http://localhost:8001/health`
- Readiness: `http://localhost:8000/ready`

4. Send an embedding request to the gateway:

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "x-api-key: dev-secret-key" \
  -d '{"input":"hello world","model":"all-MiniLM-L6-v2"}'
```

## Docker Compose details

The Compose file builds two images:

- `model-server`
- `gateway`

The `nginx` service uses the official `nginx:alpine` image and does not build from source.

### Important environment variables

- `DOCKER_REGISTRY` (optional): registry host, e.g. `docker.io`.
- `IMAGE_PREFIX`: prefix used for published image names.
- `TAG`: image tag computed by the workflow.
- `MODEL_PATH`: path to the local model directory.
- `API_KEY`: gateway authentication key.
- `MODEL_SERVER_URL`: URL of the model server used by the gateway.

## GitHub Actions Docker publish workflow

The workflow at `.github/workflows/docker-publish.yml` builds and pushes Docker images for the Compose services to a registry.

### Required repository secrets

- `DOCKER_USERNAME`
- `DOCKER_PASSWORD`

### Optional repository secret

- `DOCKER_REGISTRY` (defaults to `docker.io` if not provided)

### Behavior

- On pushes to `main`
- On pushes to tags matching `v*`
- Manual workflow dispatch

The workflow builds and pushes the `model-server` and `gateway` Compose services.

## Production notes

- `model-server` loads the model from `MODEL_PATH` and uses local files only, so you must mount or copy the model artifacts into the container.
- The gateway validates `x-api-key` against the `API_KEY` environment variable.
- NGINX is configured as a simple proxy and is optional for development.

## File structure

- `docker-compose.yml`
- `.github/workflows/docker-publish.yml`
- `gateway/Dockerfile`
- `gateway/requirements.txt`
- `gateway/app/main.py`
- `model-server/Dockerfile`
- `model-server/requirements.txt`
- `model-server/serve.py`
- `nginx/nginx.conf`

## Notes

- If you use Docker Hub, set `DOCKER_REGISTRY` to `docker.io` or leave it unset.
- If you use a private registry, set `DOCKER_REGISTRY` to the registry hostname.
- Do not store `TAG` or `IMAGE_PREFIX` as GitHub secrets unless you need a fixed tag.

## License

This repository does not include a license file. Add one if you want to publish the project publicly.
