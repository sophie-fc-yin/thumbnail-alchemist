# IAM Configuration for Thumbnail Alchemist
# Defines service accounts and IAM roles for Cloud Run, Cloud Build, and GCS access

# Grant Cloud Build service account permissions to build and deploy
resource "google_project_iam_member" "cloudbuild_run_admin" {
  project = var.project_id
  role    = "roles/run.admin"
  member  = "serviceAccount:${var.project_number}@cloudbuild.gserviceaccount.com"
}

# Grant compute service account storage admin for GCS uploads
resource "google_project_iam_member" "compute_storage_admin" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${var.compute_service_account}"
}

# Grant Cloud Build service account the ability to act as the compute service account
resource "google_service_account_iam_member" "cloudbuild_service_account_user" {
  service_account_id = "projects/${var.project_id}/serviceAccounts/${var.compute_service_account}"
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${var.project_number}@cloudbuild.gserviceaccount.com"
}

# Grant compute service account permission to sign URLs as itself
resource "google_service_account_iam_member" "compute_token_creator" {
  service_account_id = "projects/${var.project_id}/serviceAccounts/${var.compute_service_account}"
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = "serviceAccount:${var.compute_service_account}"
}

# Output IAM configuration summary
output "iam_summary" {
  description = "Summary of IAM permissions configured"
  value = {
    cloudbuild_role   = "roles/run.admin"
    compute_role      = "roles/storage.admin"
    compute_sa        = var.compute_service_account
    cloudbuild_sa     = "${var.project_number}@cloudbuild.gserviceaccount.com"
  }
}
