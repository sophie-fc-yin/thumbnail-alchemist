# Terraform Variables - Production Configuration
# This file contains the actual values for the production environment
# Values match current Cloud Run deployment

# Project Configuration
project_id     = "upbeat-cosine-335616"
project_number = "90067411133"
region         = "us-west1"
environment    = "prod"

# Cloud Run Configuration
service_name               = "thumbnail-alchemist"
compute_service_account    = "90067411133-compute@developer.gserviceaccount.com"
request_timeout_seconds    = 900  # 15 minutes
cpu_limit                  = "2"  # 2 vCPUs
memory_limit               = "4Gi" # 4 GB
min_instances              = 0    # Scale to zero
max_instances              = 10

# Storage Configuration
storage_bucket_name = "clickmoment-prod-assets"
storage_location    = "US-WEST1"  # Matches existing bucket location (cannot be changed)
enable_versioning   = false

# CORS Configuration
cors_allowed_origins = [
  "https://clickmoment.vercel.app",
  "http://localhost:3000",
  "http://localhost:8000",
]

# API Keys are managed via Google Cloud Secret Manager
# See terraform/iam.tf for secret access permissions
