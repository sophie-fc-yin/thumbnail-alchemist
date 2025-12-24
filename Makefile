# Configuration
REGISTRY := us-west1-docker.pkg.dev/upbeat-cosine-335616/thumbnail-alchemist
IMAGE_NAME := thumbnail-alchemist-api
TAG ?= latest
FULL_IMAGE := $(REGISTRY)/$(IMAGE_NAME):$(TAG)

# Cloud Run configuration
SERVICE_NAME := thumbnail-alchemist
REGION := us-west1

.PHONY: setup-env setup-dev serve test docker-build docker-push docker-build-push deploy

# Setup development environment
setup-env:
	@echo "Setting up development environment..."
	@command -v uv >/dev/null 2>&1 || { echo "Installing uv..."; curl -LsSf https://astral.sh/uv/install.sh | sh; }
	uv venv
	uv sync
	@echo "✓ Environment setup complete! Activate with: source .venv/bin/activate"

# Setup with dev dependencies and pre-commit hooks
setup-dev: setup-env
	uv sync --extra dev
	uv run pre-commit install
	@echo "✓ Dev environment ready with pre-commit hooks installed"

# Local development (hot reload, no Docker)
serve:
	uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 9000

test:
	uv run pytest

# Build Docker image for AMD64 (Cloud Run compatible)
docker-build:
	docker build --platform linux/amd64 -t $(FULL_IMAGE) .
	@echo "Built: $(FULL_IMAGE)"

# Push Docker image to Google Artifact Registry
docker-push:
	docker push $(FULL_IMAGE)
	@echo "Pushed: $(FULL_IMAGE)"

# Build and push in one command
docker-build-push: docker-build docker-push

# Build, push, and deploy to production
deploy: docker-build-push
	@echo "Deploying latest image to Cloud Run..."
	@DIGEST=$$(gcloud artifacts docker images describe $(FULL_IMAGE) --format='get(image_summary.digest)' 2>/dev/null || echo ""); \
	if [ -z "$$DIGEST" ]; then \
		echo "Warning: Could not get image digest, deploying with tag..."; \
		IMAGE_REF=$(FULL_IMAGE); \
	else \
		IMAGE_REF=$(REGISTRY)/$(IMAGE_NAME)@$$DIGEST; \
		echo "Using digest: $$DIGEST"; \
	fi; \
	gcloud run deploy $(SERVICE_NAME) \
		--image $$IMAGE_REF \
		--region $(REGION) \
		--platform managed \
		--allow-unauthenticated
	@echo "✓ Deployed successfully"
