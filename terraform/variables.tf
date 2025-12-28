# Variables for Thumbnail Alchemist Infrastructure
# Define all configurable parameters for the infrastructure

# ============================================================================
# Project Configuration
# ============================================================================

variable "project_id" {
  description = "Google Cloud project ID"
  type        = string
}

variable "project_number" {
  description = "Google Cloud project number"
  type        = string
}

variable "region" {
  description = "Google Cloud region for Cloud Run deployment"
  type        = string
  default     = "us-west1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "prod"
}

# ============================================================================
# Cloud Run Configuration
# ============================================================================

variable "service_name" {
  description = "Name of the Cloud Run service"
  type        = string
  default     = "thumbnail-alchemist"
}

variable "container_image" {
  description = "Container image for Cloud Run (e.g., from Artifact Registry or GCR)"
  type        = string
  # This will be updated by Cloud Build when deploying from source
  default     = "us-west1-docker.pkg.dev/upbeat-cosine-335616/cloud-run-source-deploy/thumbnail-alchemist"
}

variable "compute_service_account" {
  description = "Service account for Cloud Run compute"
  type        = string
}

variable "request_timeout_seconds" {
  description = "Cloud Run request timeout in seconds (max 3600 = 60 minutes)"
  type        = number
  default     = 900 # 15 minutes
}

variable "cpu_limit" {
  description = "CPU limit for Cloud Run containers"
  type        = string
  default     = "2" # 2 vCPUs
}

variable "memory_limit" {
  description = "Memory limit for Cloud Run containers"
  type        = string
  default     = "4Gi" # 4 GB
}

variable "min_instances" {
  description = "Minimum number of Cloud Run instances"
  type        = number
  default     = 0 # Scale to zero when not in use
}

variable "max_instances" {
  description = "Maximum number of Cloud Run instances"
  type        = number
  default     = 10
}

# ============================================================================
# Storage Configuration
# ============================================================================

variable "storage_bucket_name" {
  description = "Name of the GCS bucket for storing assets"
  type        = string
  default     = "clickmoment-prod-assets"
}

variable "storage_location" {
  description = "Location for GCS bucket (multi-region or region)"
  type        = string
  default     = "US" # Multi-region for better availability
}

variable "enable_versioning" {
  description = "Enable versioning for GCS bucket"
  type        = bool
  default     = false
}

variable "cors_allowed_origins" {
  description = "List of allowed origins for CORS on GCS bucket"
  type        = list(string)
  default = [
    "https://clickmoment.vercel.app",
    "http://localhost:3000",
  ]
}
