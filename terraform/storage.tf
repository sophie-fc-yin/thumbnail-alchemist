# Google Cloud Storage Configuration
# Defines the GCS bucket for storing uploaded videos, processed frames, and audio files

# Main storage bucket for production assets
resource "google_storage_bucket" "clickmoment_prod_assets" {
  name          = var.storage_bucket_name
  location      = var.storage_location
  force_destroy = false # Prevent accidental deletion of production data

  # Uniform bucket-level access (recommended for new buckets)
  uniform_bucket_level_access = true

  # CORS configuration to allow uploads from frontend
  cors {
    origin = var.cors_allowed_origins
    method = ["GET", "HEAD", "PUT", "POST", "DELETE", "OPTIONS"]
    response_header = [
      "Content-Type",
      "Content-Length",
      "Content-MD5",
      "x-goog-acl",
      "x-goog-content-length-range",
      "x-goog-meta-*",
      "Authorization",
      "Access-Control-Allow-Origin",
      "Access-Control-Allow-Methods",
      "Access-Control-Allow-Headers"
    ]
    max_age_seconds = 3600
  }

  # Lifecycle rules for cost optimization (optional)
  lifecycle_rule {
    condition {
      age = 90 # Delete files older than 90 days
    }
    action {
      type = "Delete"
    }
  }

  # Enable versioning for important data (optional)
  versioning {
    enabled = var.enable_versioning
  }

  labels = {
    environment = var.environment
    managed-by  = "terraform"
  }
}

# Grant Cloud Run service account access to the bucket
resource "google_storage_bucket_iam_member" "cloudrun_storage_access" {
  bucket = google_storage_bucket.clickmoment_prod_assets.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${var.compute_service_account}"
}

# Output bucket information
output "storage_bucket_name" {
  description = "Name of the GCS bucket"
  value       = google_storage_bucket.clickmoment_prod_assets.name
}

output "storage_bucket_url" {
  description = "GCS URL of the bucket"
  value       = "gs://${google_storage_bucket.clickmoment_prod_assets.name}"
}
