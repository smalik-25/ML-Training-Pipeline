variable "aws_region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "us-east-1"
}

variable "project" {
  description = "Name prefix for resources."
  type        = string
  default     = "sneaker-ml"
}

variable "bucket_name" {
  description = "Globally-unique S3 bucket name for the data lake."
  type        = string
}

variable "image_tag" {
  description = "ECR image tag App Runner serves."
  type        = string
  default     = "latest"
}

variable "model_uri" {
  description = <<-EOT
    s3:// path to the model.pt the serving app loads at startup, e.g.
    s3://<bucket>/models/2026-06-28/<run_id>/model.pt. Leave empty on the first
    apply (before a model is uploaded); set it before deploy_service = true.
  EOT
  type        = string
  default     = ""
}

variable "deploy_service" {
  description = <<-EOT
    Whether to create the App Runner service. Apply once with false to create the
    bucket, ECR repo, and IAM; push the image and upload a model; then set true.
  EOT
  type        = bool
  default     = false
}

variable "service_cpu" {
  description = "App Runner vCPU (1024 = 1 vCPU). torch needs headroom."
  type        = string
  default     = "1024"
}

variable "service_memory" {
  description = "App Runner memory in MB."
  type        = string
  default     = "2048"
}
