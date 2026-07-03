output "bucket_name" {
  description = "Data lake bucket. Use as storage_root: s3://<bucket_name>."
  value       = aws_s3_bucket.lake.bucket
}

output "ecr_repository_url" {
  description = "Push the serving image here."
  value       = aws_ecr_repository.serving.repository_url
}

output "service_url" {
  description = "App Runner HTTPS URL (null until deploy_service = true)."
  value       = try(aws_apprunner_service.serving[0].service_url, null)
}
