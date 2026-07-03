# Deploying the serving API to AWS App Runner

This deploys the FastAPI price-premium service to AWS: an S3 data-lake bucket, an
ECR repository, and an App Runner service that loads the model from S3 and serves
it over HTTPS with autoscaling. The batch pipeline can also run against the same
bucket by setting `storage_root` to `s3://<bucket>`.

> **Cost and teardown.** This creates billable resources. App Runner keeps at
> least one instance warm, so it is not free while it exists (roughly
> **$5–45/month** for a 1 vCPU / 2 GB service depending on traffic; S3 and ECR
> are pennies). **Run `terraform destroy` when you're done** (see Teardown).

## Prerequisites

- An AWS account and the AWS CLI configured (`aws configure`, or SSO). Verify
  with `aws sts get-caller-identity`.
- Terraform >= 1.5 and Docker installed locally.
- A model artifact to serve. The fast path uses the `model.pt` from your local
  run; the full path runs the pipeline against S3 (see step 2).

## 1. Provision storage + registry + IAM (no service yet)

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars
# edit terraform.tfvars: set a globally-unique bucket_name (leave deploy_service unset)

terraform init
terraform validate
terraform apply          # creates the bucket, ECR repo, and IAM roles
```

Note the outputs: `bucket_name` and `ecr_repository_url`.

## 2. Put a model in the lake

**Fast path** — upload the `model.pt` from your local run:

```bash
# find your run_id under data/models/<run_date>/
aws s3 cp "data/models/2026-06-28/<run_id>/model.pt" \
  "s3://<bucket_name>/models/2026-06-28/<run_id>/model.pt"
```

**Full path** — run the pipeline against S3 so it lands there natively:

```bash
export STORAGE_ROOT="s3://<bucket_name>"   # overrides storage_root in the config
export SNEAKER_INTEL_DSN=...               # if ingesting from the real warehouse
python -m stages.ingest   --run-date 2026-06-28
python -m stages.features --run-date 2026-06-28
python -m stages.validate --run-date 2026-06-28
python -m stages.train    --run-date 2026-06-28 --split-year 2019
python -m stages.register --run-date 2026-06-28
```

Either way, note the full `s3://.../model.pt` path; that's your `model_uri`.

## 3. Build and push the serving image to ECR

Locally:

```bash
aws ecr get-login-password --region <region> \
  | docker login --username AWS --password-stdin <ecr_repository_url>
docker build -f serving/Dockerfile -t <ecr_repository_url>:latest .
docker push <ecr_repository_url>:latest
```

Or via GitHub Actions: set repo secrets `AWS_ACCESS_KEY_ID` /
`AWS_SECRET_ACCESS_KEY` and repo vars `AWS_REGION` / `ECR_REPOSITORY`, then run
the **deploy-serving** workflow (or push a `v*` tag). It builds and pushes the
image; because the service has auto-deploy on, a push also redeploys it.

## 4. Create the App Runner service

Set the remaining vars in `terraform.tfvars`:

```hcl
deploy_service = true
image_tag      = "latest"
model_uri      = "s3://<bucket_name>/models/2026-06-28/<run_id>/model.pt"
```

```bash
terraform apply
```

Grab the URL:

```bash
terraform output service_url
```

## 5. Test it

```bash
URL="https://<service_url>"
curl -s "$URL/health" | python -m json.tool
curl -s -X POST "$URL/predict" -H "Content-Type: application/json" \
  -d '{"days_since_release":100,"size_us":9.0,"retail_price":180,"size_premium":0.1,"release_type_encoded":2,"brand_avg_premium":0.5}'
```

`/health` reports the loaded model version; `/predict` returns a premium. The
model is read from S3 using the App Runner instance role, so no keys live in the
image.

## Updating and rolling back

- **New model, same image:** upload the new `model.pt`, set `model_uri` in
  `terraform.tfvars`, `terraform apply`. (The env change redeploys the service;
  or, if the service is up, call `POST /reload` after changing `MODEL_URI`.)
- **New code:** push a new image to ECR; auto-deploy rolls it out.

## Teardown

Versioned buckets must be emptied before they can be deleted:

```bash
aws s3 rm "s3://<bucket_name>" --recursive
# also remove old versions if any:
# aws s3api delete-objects ... (or empty via the console)

cd infra/terraform
terraform destroy
```

Confirm in the AWS console that the App Runner service, ECR repo, and bucket are
gone so nothing keeps billing.

## Security notes

- The serving instance role is least-privilege: `s3:GetObject` on
  `models/*` and `predictions/*` only, no write, no other buckets.
- No credentials are baked into the image; App Runner injects the instance role.
- For CI, prefer GitHub OIDC (an assumable role) over long-lived access keys.
- The bucket blocks all public access and is encrypted at rest.
