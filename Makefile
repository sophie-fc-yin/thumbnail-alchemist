# Configuration
REGISTRY := us-west1-docker.pkg.dev/upbeat-cosine-335616/thumbnail-alchemist
IMAGE_NAME := thumbnail-alchemist-api
TAG ?= latest
FULL_IMAGE := $(REGISTRY)/$(IMAGE_NAME):$(TAG)

# Cloud Run configuration
SERVICE_NAME := thumbnail-alchemist
REGION := us-west1

.PHONY: setup-env setup-dev serve test docker-build docker-push docker-build-push deploy terraform-init terraform-plan terraform-apply terraform-apply-auto deploy-full

# Setup development environment
setup-env:
	@echo "Setting up development environment..."
	@echo "Checking system dependencies..."
	@command -v ffmpeg >/dev/null 2>&1 || { echo "❌ FFmpeg not found! Install with: brew install ffmpeg"; exit 1; }
	@echo "✓ FFmpeg found: $$(ffmpeg -version | head -1)"
	@command -v uv >/dev/null 2>&1 || { echo "Installing uv..."; curl -LsSf https://astral.sh/uv/install.sh | sh; }
	uv venv
	uv sync
	@echo "✓ Environment setup complete! Activate with: source .venv/bin/activate"

# Setup with dev dependencies and pre-commit hooks
setup-dev: setup-env
	uv sync --extra dev
	uv run pre-commit install
	@echo "Pre-downloading ML models..."
	@uv run python scripts/download_models.py
	@echo "✓ Dev environment ready with pre-commit hooks installed"

# Local development (hot reload, no Docker)
serve:
	@source ~/.zshrc && DYLD_LIBRARY_PATH=/opt/homebrew/lib:$$DYLD_LIBRARY_PATH uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 9000

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

# Terraform targets for infrastructure management
terraform-init:
	@echo "Initializing Terraform..."
	cd terraform && terraform init

terraform-plan:
	@echo "Planning Terraform changes..."
	cd terraform && terraform plan

terraform-apply:
	@echo "Applying Terraform changes..."
	cd terraform && terraform apply

terraform-apply-auto:
	@echo "Applying Terraform changes (auto-approve)..."
	cd terraform && terraform apply -auto-approve

# Build, push, and deploy to production
# NOTE: This does NOT apply Terraform changes.
#
# IMPORTANT: If Terraform manages your Cloud Run service (see terraform/main.tf),
# using 'gcloud run deploy' may override Terraform configuration (env vars, secrets, etc.).
# For infrastructure changes (IAM, secrets, env vars), use 'make terraform-apply' instead.
#
# For code-only updates (no infrastructure changes), this command is safe to use.
deploy: docker-build-push
	@echo "⚠️  WARNING: This uses 'gcloud run deploy' which may conflict with Terraform-managed resources."
	@echo "If you've updated Terraform config (secrets, env vars, IAM), run 'make terraform-apply' first."
	@echo ""
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

# Full deployment: apply Terraform changes, then build and deploy container
# WARNING: This will run terraform apply FIRST, then deploy.
# Terraform manages the Cloud Run service, so the gcloud deploy may override terraform settings.
# For infrastructure updates (secrets, IAM, env vars), use 'make terraform-apply' instead.
# For code updates only, use 'make deploy'.
deploy-full: terraform-apply docker-build-push
	@echo "⚠️  WARNING: Terraform manages the Cloud Run service."
	@echo "Running 'gcloud run deploy' after terraform apply may override terraform configuration."
	@echo "Consider updating terraform.tfvars container_image variable and using terraform apply instead."
	@echo ""
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
	@echo "✓ Full deployment complete (Terraform + container)"
