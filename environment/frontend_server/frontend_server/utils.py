from __future__ import annotations

from storages.backends.s3boto import S3BotoStorage


StaticRootS3BotoStorage = lambda: S3BotoStorage(location="static")
MediaRootS3BotoStorage = lambda: S3BotoStorage(location="media")
