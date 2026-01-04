# Thumbnail Alchemist - Terraform Infrastructure

This directory contains the Terraform configuration for managing the Thumbnail Alchemist infrastructure on Google Cloud Platform.

## 📋 What's Managed by Terraform

This Terraform configuration manages:

- **Cloud Run Service** (`main.tf`)
  - Service deployment with timeout, CPU, and memory settings
  - Auto-scaling configuration
  - Public access permissions

- **IAM Permissions** (`iam.tf`)
  - Cloud Build service account permissions
  - Cloud Run compute service account permissions
  - Storage access permissions

- **Google Cloud Storage** (`storage.tf`)
  - GCS bucket for video uploads, frames, and audio files
  - CORS configuration for frontend uploads
  - Lifecycle rules for cost optimization

## 🏗️ Infrastructure Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Cloud Run Service                        │
│  thumbnail-alchemist (us-west1)                             │
│  - FastAPI backend                                          │
│  - 15min timeout (large video uploads)                      │
│  - 2 vCPU, 4GB memory                                       │
│  - Auto-scales 0-10 instances                               │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Google Cloud Storage (GCS)                      │
│  clickmoment-prod-assets                                    │
│  - users/{user_id}/videos/                                  │
│  - projects/{project_id}/signals/frames/                    │
│  - projects/{project_id}/signals/audio/                     │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Prerequisites

1. **Install Terraform** (v1.0 or higher):
```bash
# macOS
brew install terraform

# Or download from https://www.terraform.io/downloads
```

2. **Google Cloud Authentication**:
```bash
# Login to gcloud
gcloud auth login

# Set application default credentials
gcloud auth application-default login

# Set your project
gcloud config set project upbeat-cosine-335616
```

3. **Enable Required APIs** (if not already enabled):
```bash
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable artifactregistry.googleapis.com
```

## 📦 Current Production Configuration

The `terraform.tfvars` file contains the current production configuration that matches what's deployed:

| Setting | Value | Description |
|---------|-------|-------------|
| **Project ID** | `upbeat-cosine-335616` | GCP project |
| **Region** | `us-west1` | Cloud Run region |
| **Service Name** | `thumbnail-alchemist` | Cloud Run service name |
| **Timeout** | `900s` (15 min) | Request timeout |
| **CPU** | `2` vCPUs | Container CPU limit |
| **Memory** | `4Gi` | Container memory limit |
| **Bucket** | `clickmoment-prod-assets` | GCS bucket name |

### 🔑 Required API Keys (via Secret Manager)

The application requires two API keys stored in **Google Cloud Secret Manager**:

1. **OPENAI_API_KEY** - For GPT-4o Transcribe Diarize (audio transcription and analysis)
2. **GEMINI_API_KEY** - For Gemini 2.5 Flash (AI-powered thumbnail selection)

#### Create Secrets in Secret Manager

You can create/update secrets via the [Google Cloud Console](https://console.cloud.google.com/security/secret-manager) or using `gcloud`:

```bash
# Create OPENAI_API_KEY secret (if not exists)
echo -n "sk-proj-your-openai-key" | gcloud secrets create OPENAI_API_KEY \
  --data-file=- \
  --replication-policy="automatic"

# Or update existing secret with new version
echo -n "sk-proj-your-openai-key" | gcloud secrets versions add OPENAI_API_KEY \
  --data-file=-

# Create GEMINI_API_KEY secret (if not exists)
echo -n "AIza-your-gemini-key" | gcloud secrets create GEMINI_API_KEY \
  --data-file=- \
  --replication-policy="automatic"

# Or update existing secret with new version
echo -n "AIza-your-gemini-key" | gcloud secrets versions add GEMINI_API_KEY \
  --data-file=-
```

#### Verify Secrets

List all secrets:
```bash
gcloud secrets list
```

View secret details (not the actual value):
```bash
gcloud secrets describe OPENAI_API_KEY
gcloud secrets describe GEMINI_API_KEY
```

The Terraform configuration automatically:
- Grants the Cloud Run service account `secretmanager.secretAccessor` role
- Mounts the secrets as environment variables in the container
- Uses the `latest` version of each secret

## 🛠️ Usage

### Set API Keys in Secret Manager (One-Time Setup)

Before deploying for the first time, create the required secrets:

```bash
# Create GEMINI_API_KEY if it doesn't exist
echo -n "AIza-your-gemini-key" | gcloud secrets create GEMINI_API_KEY \
  --data-file=- \
  --replication-policy="automatic"

# OPENAI_API_KEY should already exist (visible in your screenshot)
# If you need to update it:
echo -n "sk-proj-your-openai-key" | gcloud secrets versions add OPENAI_API_KEY \
  --data-file=-
```

### Initialize Terraform

Run this first time or after adding new providers:

```bash
cd terraform
terraform init
```

### Preview Changes

See what changes Terraform will make:

```bash
terraform plan
```

This shows:
- Resources to be created (+)
- Resources to be modified (~)
- Resources to be deleted (-)

### Apply Changes

Apply the infrastructure changes:

```bash
terraform apply
```

Terraform will:
1. Show you a plan
2. Ask for confirmation (type `yes`)
3. Create/update resources (including Secret Manager IAM permissions)
4. Mount secrets as environment variables in Cloud Run
5. Save state to `terraform.tfstate`

### View Current State

See what resources are managed:

```bash
terraform show
```

### Destroy Infrastructure

⚠️ **DANGER**: This will delete all managed resources!

```bash
terraform destroy
```

## 📝 Making Changes

### Update Cloud Run Timeout

Edit `terraform.tfvars`:
```hcl
request_timeout_seconds = 1800  # 30 minutes
```

Then apply:
```bash
terraform apply
```

### Add Environment Variables

Edit `main.tf` in the `containers` block:
```hcl
containers {
  image = var.container_image

  env {
    name  = "ENVIRONMENT"
    value = var.environment
  }

  env {
    name  = "DEBUG"
    value = "false"
  }
}
```

### Change Resource Limits

Edit `terraform.tfvars`:
```hcl
cpu_limit    = "4"    # 4 vCPUs
memory_limit = "8Gi"  # 8 GB
```

## 🔄 Importing Existing Resources

If you already have resources in Google Cloud and want Terraform to manage them, use `terraform import`:

```bash
# Import existing Cloud Run service
terraform import google_cloud_run_v2_service.thumbnail_alchemist \
  projects/upbeat-cosine-335616/locations/us-west1/services/thumbnail-alchemist

# Import existing GCS bucket
terraform import google_storage_bucket.clickmoment_prod_assets clickmoment-prod-assets
```

## 🔒 State Management

### Current Setup

Terraform state is stored locally in `terraform.tfstate` (gitignored). This file contains:
- Current infrastructure state
- Resource IDs and metadata
- Sensitive information

⚠️ **Important**: Never commit `terraform.tfstate` to git!

### Recommended: Remote State (Production)

For production, use Google Cloud Storage for remote state:

Create a backend configuration file `backend.tf`:
```hcl
terraform {
  backend "gcs" {
    bucket = "thumbnail-alchemist-terraform-state"
    prefix = "prod"
  }
}
```

Create the state bucket:
```bash
gsutil mb gs://thumbnail-alchemist-terraform-state
gsutil versioning set on gs://thumbnail-alchemist-terraform-state
```

Initialize with remote backend:
```bash
terraform init -migrate-state
```

## 🔍 Troubleshooting

### "Resource already exists" error

If a resource already exists, import it first:
```bash
terraform import <resource_type>.<resource_name> <resource_id>
```

### Permission denied errors

Ensure your service account has the required roles:
```bash
# Check your current account
gcloud auth list

# Grant necessary permissions
gcloud projects add-iam-policy-binding upbeat-cosine-335616 \
  --member="user:your-email@example.com" \
  --role="roles/editor"
```

### State lock errors

If Terraform is stuck with a state lock:
```bash
terraform force-unlock <lock_id>
```

## 📚 Files Overview

```
terraform/
├── README.md              # This file
├── .gitignore            # Ignore state and sensitive files
├── main.tf               # Cloud Run service definition
├── iam.tf                # IAM permissions and service accounts
├── storage.tf            # GCS bucket configuration
├── variables.tf          # Variable definitions
├── terraform.tfvars      # Production values (committed)
└── terraform.tfstate     # State file (gitignored, auto-generated)
```

## 🎯 Best Practices

1. **Always run `terraform plan` first** - Review changes before applying
2. **Use version control** - Commit all `.tf` files (except state)
3. **Document changes** - Add comments explaining why changes were made
4. **Use variables** - Keep configuration flexible and reusable
5. **Enable remote state** - For team collaboration
6. **Use workspaces** - For multiple environments (dev, staging, prod)

## 🔗 Related Documentation

- [Terraform Google Provider](https://registry.terraform.io/providers/hashicorp/google/latest/docs)
- [Cloud Run Terraform Resource](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/cloud_run_v2_service)
- [GCS Terraform Resource](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/storage_bucket)
- [Main Project README](../README.md)

## ⚡ Quick Reference

```bash
# Initialize
terraform init

# Preview changes
terraform plan

# Apply changes
terraform apply

# Apply without confirmation (use with caution!)
terraform apply -auto-approve

# View outputs
terraform output

# Format code
terraform fmt

# Validate configuration
terraform validate

# Show current state
terraform show

# List managed resources
terraform state list
```
