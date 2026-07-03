locals {
  tags = {
    Project   = var.project
    ManagedBy = "terraform"
  }
}

# ---------------------------------------------------------------------------
# Data lake bucket. storage_root points here (s3://<bucket>). Private,
# versioned, encrypted.
# ---------------------------------------------------------------------------
resource "aws_s3_bucket" "lake" {
  bucket = var.bucket_name
  tags   = local.tags
}

resource "aws_s3_bucket_versioning" "lake" {
  bucket = aws_s3_bucket.lake.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "lake" {
  bucket                  = aws_s3_bucket.lake.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "lake" {
  bucket = aws_s3_bucket.lake.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# ---------------------------------------------------------------------------
# ECR repository for the serving image.
# ---------------------------------------------------------------------------
resource "aws_ecr_repository" "serving" {
  name                 = "${var.project}-serving"
  image_tag_mutability = "MUTABLE"
  force_delete         = true
  image_scanning_configuration {
    scan_on_push = true
  }
  tags = local.tags
}

# ---------------------------------------------------------------------------
# IAM. Two roles: one lets App Runner pull from ECR; one is the running
# service's identity, scoped to read models/predictions from the lake.
# ---------------------------------------------------------------------------
data "aws_iam_policy_document" "apprunner_ecr_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["build.apprunner.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "apprunner_ecr_access" {
  name               = "${var.project}-apprunner-ecr-access"
  assume_role_policy = data.aws_iam_policy_document.apprunner_ecr_assume.json
  tags               = local.tags
}

resource "aws_iam_role_policy_attachment" "apprunner_ecr_access" {
  role       = aws_iam_role.apprunner_ecr_access.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
}

data "aws_iam_policy_document" "apprunner_instance_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["tasks.apprunner.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "s3_read" {
  statement {
    sid       = "ReadModelArtifacts"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.lake.arn}/models/*", "${aws_s3_bucket.lake.arn}/predictions/*"]
  }
  statement {
    sid       = "ListLakePrefixes"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.lake.arn]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["models/*", "predictions/*"]
    }
  }
}

resource "aws_iam_role" "apprunner_instance" {
  name               = "${var.project}-apprunner-instance"
  assume_role_policy = data.aws_iam_policy_document.apprunner_instance_assume.json
  tags               = local.tags
}

resource "aws_iam_role_policy" "apprunner_instance_s3" {
  name   = "s3-read"
  role   = aws_iam_role.apprunner_instance.id
  policy = data.aws_iam_policy_document.s3_read.json
}

# ---------------------------------------------------------------------------
# App Runner service. count = 0 until the image is pushed and a model uploaded
# (deploy_service = true). Auto-deploys on a new image push to ECR.
# ---------------------------------------------------------------------------
resource "aws_apprunner_service" "serving" {
  count        = var.deploy_service ? 1 : 0
  service_name = "${var.project}-serving"

  source_configuration {
    auto_deployments_enabled = true
    authentication_configuration {
      access_role_arn = aws_iam_role.apprunner_ecr_access.arn
    }
    image_repository {
      image_identifier      = "${aws_ecr_repository.serving.repository_url}:${var.image_tag}"
      image_repository_type = "ECR"
      image_configuration {
        port = "8080"
        runtime_environment_variables = {
          MODEL_URI          = var.model_uri
          AWS_DEFAULT_REGION = var.aws_region
        }
      }
    }
  }

  instance_configuration {
    cpu               = var.service_cpu
    memory            = var.service_memory
    instance_role_arn = aws_iam_role.apprunner_instance.arn
  }

  health_check_configuration {
    protocol            = "HTTP"
    path                = "/health"
    interval            = 10
    timeout             = 5
    healthy_threshold   = 1
    unhealthy_threshold = 5
  }

  tags = local.tags
}
