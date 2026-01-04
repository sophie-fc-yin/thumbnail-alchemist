# Thumbnail Alchemist - Cloud Run Service Configuration
# This file defines the Cloud Run service that hosts the FastAPI backend

terraform {
  required_version = ">= 1.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Cloud Run Service
resource "google_cloud_run_v2_service" "thumbnail_alchemist" {
  name     = var.service_name
  location = var.region

  template {
    # Service account for the Cloud Run service
    service_account = var.compute_service_account

    # Timeout configuration - allows time for large video uploads
    timeout = "${var.request_timeout_seconds}s"

    containers {
      # Container image - built from source via Cloud Build
      image = var.container_image

      # Resource limits
      resources {
        limits = {
          cpu    = var.cpu_limit
          memory = var.memory_limit
        }
      }

      # Environment variables
      env {
        name  = "ENVIRONMENT"
        value = var.environment
      }

      # Suppress PyTorch NNPACK warnings
      env {
        name  = "PYTORCH_DISABLE_NNPACK"
        value = "1"
      }
      env {
        name  = "TORCH_NNPACK_DISABLE"
        value = "1"
      }
      env {
        name  = "OMP_NUM_THREADS"
        value = "1"
      }

      # Mount secrets from Secret Manager
      env {
        name = "OPENAI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = "OPENAI_API_KEY"
            version = "latest"
          }
        }
      }

      env {
        name = "GEMINI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = "GEMINI_API_KEY"
            version = "latest"
          }
        }
      }
    }

    # Scaling configuration
    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }
  }

  # Traffic configuration - 100% to latest revision
  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }
}

# Allow unauthenticated access (public API)
resource "google_cloud_run_v2_service_iam_member" "public_access" {
  location = google_cloud_run_v2_service.thumbnail_alchemist.location
  name     = google_cloud_run_v2_service.thumbnail_alchemist.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Output the service URL
output "service_url" {
  description = "URL of the deployed Cloud Run service"
  value       = google_cloud_run_v2_service.thumbnail_alchemist.uri
}

output "service_name" {
  description = "Name of the Cloud Run service"
  value       = google_cloud_run_v2_service.thumbnail_alchemist.name
}
