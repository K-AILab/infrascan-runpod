"""Stage B — object storage I/O for the serverless GPU worker.

Serverless workers are stateless/ephemeral, and the full-pipeline output (posed
dataset + DA3 depth npz + object point cloud + index) is far too big to return
as base64. So the worker uploads a result archive to an S3-compatible bucket
(AWS S3 or Cloudflare R2) and returns a URL the frontend downloads.

Config via env (set these as RunPod endpoint secrets):
    S3_ENDPOINT_URL   e.g. https://<accountid>.r2.cloudflarestorage.com   (omit for AWS S3)
    S3_BUCKET         bucket name
    S3_ACCESS_KEY     access key id
    S3_SECRET_KEY     secret access key
    S3_REGION         region (default "auto" for R2, "us-east-1" for S3)
    S3_PUBLIC_BASE    optional public base URL for GET (e.g. an R2 public bucket / CDN)

Falls back cleanly: if no bucket is configured, upload() raises with a clear
message so the handler can report it instead of silently losing the result.
"""
from __future__ import annotations

import os
from pathlib import Path


def _client():
    import boto3  # imported lazily so the module loads even without boto3 present
    from botocore.config import Config

    endpoint = os.environ.get("S3_ENDPOINT_URL") or None
    region = os.environ.get("S3_REGION") or ("auto" if endpoint else "us-east-1")
    key = os.environ.get("S3_ACCESS_KEY")
    secret = os.environ.get("S3_SECRET_KEY")
    if not (os.environ.get("S3_BUCKET") and key and secret):
        raise RuntimeError(
            "storage not configured: set S3_BUCKET, S3_ACCESS_KEY, S3_SECRET_KEY "
            "(and S3_ENDPOINT_URL for Cloudflare R2) as endpoint secrets."
        )
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        config=Config(signature_version="s3v4", retries={"max_attempts": 3}),
    )


def upload(local_path: str | Path, key: str) -> str:
    """Upload a file to the bucket under `key`. Returns a fetchable URL.

    If S3_PUBLIC_BASE is set, returns that public URL; otherwise returns a
    presigned GET URL valid for 24h.
    """
    local_path = str(local_path)
    bucket = os.environ["S3_BUCKET"]
    c = _client()
    c.upload_file(local_path, bucket, key)

    public_base = os.environ.get("S3_PUBLIC_BASE")
    if public_base:
        return f"{public_base.rstrip('/')}/{key}"
    return c.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=24 * 3600
    )


def download(url_or_key: str, dest: str | Path) -> str:
    """Download an input by presigned/public URL (http) or by bucket key."""
    dest = str(dest)
    if url_or_key.startswith(("http://", "https://")):
        import urllib.request
        urllib.request.urlretrieve(url_or_key, dest)
    else:
        _client().download_file(os.environ["S3_BUCKET"], url_or_key, dest)
    return dest
