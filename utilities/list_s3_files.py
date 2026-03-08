import argparse
import json
import boto3
from typing import List, Dict, Any, Set


def list_s3_bucket_files(bucket_name: str, prefix: str = "", region: str = "us-east-1") -> List[Dict[str, Any]]:
    """
    List all files in an S3 bucket with optional prefix filtering.

    Args:
        bucket_name: Name of the S3 bucket
        prefix: Optional prefix to filter files (e.g., "embeddings/")
        region: AWS region for the bucket

    Returns:
        List of dictionaries containing file metadata
    """
    s3_client = boto3.client("s3", region_name=region)
    files = []

    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

        for page in pages:
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                files.append(
                    {
                        "bucket": bucket_name,
                        "key": obj["Key"],
                        "size_bytes": obj["Size"],
                        "last_modified": obj["LastModified"].isoformat(),
                    }
                )

        print(f"Found {len(files)} files in s3://{bucket_name}/{prefix}")
        return files

    except Exception as e:
        print(f"Error listing S3 bucket: {e}")
        raise


def print_files_table(files: List[Dict[str, Any]]) -> None:
    """
    Print files in a formatted table.

    Args:
        files: List of file metadata dictionaries
    """
    if not files:
        print("No files found.")
        return

    print("\n{:<60} {:<15} {:<20}".format("Key", "Size (bytes)", "Last Modified"))
    print("-" * 95)

    for file_info in files:
        key = file_info["key"]
        size = file_info["size_bytes"]
        modified = file_info["last_modified"][:10]  # Date only
        print("{:<60} {:<15} {:<20}".format(key[:60], str(size), modified))


def extract_source_urls_from_s3(bucket_name: str, prefix: str = "", region: str = "us-east-1") -> Dict[str, Any]:
    """
    Extract unique source URLs from embeddings JSON files in S3.

    Args:
        bucket_name: Name of the S3 bucket
        prefix: Optional prefix to filter files (e.g., "embeddings/")
        region: AWS region for the bucket

    Returns:
        Dictionary with extracted URLs and metadata
    """
    s3_client = boto3.client("s3", region_name=region)
    unique_urls: Set[str] = set()
    files_processed = 0
    urls_found = 0

    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

        for page in pages:
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                key = obj["Key"]
                if not key.endswith(".json"):
                    continue

                try:
                    response = s3_client.get_object(Bucket=bucket_name, Key=key)
                    content = response["Body"].read().decode("utf-8")
                    records = json.loads(content)

                    files_processed += 1

                    if isinstance(records, list):
                        for record in records:
                            metadata = record.get("metadata", {}) or {}
                            source = metadata.get("source") or metadata.get("source_url") or ""
                            if source:
                                unique_urls.add(source)
                                urls_found += 1

                except Exception as parse_error:
                    print(f"Warning: Could not parse {key}: {parse_error}")

        return {
            "bucket": bucket_name,
            "prefix": prefix,
            "files_processed": files_processed,
            "total_urls_found": urls_found,
            "unique_source_urls": sorted(list(unique_urls)),
        }

    except Exception as e:
        print(f"Error extracting URLs from S3: {e}")
        raise


def print_source_urls(result: Dict[str, Any]) -> None:
    """
    Print extracted source URLs in a formatted list.

    Args:
        result: Dictionary returned from extract_source_urls_from_s3
    """
    urls = result.get("unique_source_urls", [])
    if not urls:
        print("No source URLs found.")
        return

    print(f"\nExtracted Source URLs from s3://{result['bucket']}/{result['prefix']}")
    print(f"Files processed: {result['files_processed']}")
    print(f"Total URLs found: {result['total_urls_found']}")
    print(f"Unique URLs: {len(urls)}\n")
    print("-" * 80)

    for idx, url in enumerate(urls, 1):
        print(f"{idx}. {url}")


def main() -> None:
    parser = argparse.ArgumentParser(description="List files and extract source URLs from S3 bucket")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--prefix", default="", help="Optional prefix to filter files (e.g., 'embeddings/')")
    parser.add_argument("--region", default="us-east-1", help="AWS region for the bucket")
    parser.add_argument(
        "--output", choices=["table", "json", "urls"], default="table", help="Output format"
    )
    parser.add_argument(
        "--extract-urls", action="store_true", help="Extract and display source URLs from embeddings files"
    )

    args = parser.parse_args()

    if args.extract_urls:
        result = extract_source_urls_from_s3(args.bucket, args.prefix, args.region)

        if args.output == "json":
            print(json.dumps(result, indent=2))
        else:
            print_source_urls(result)
    else:
        files = list_s3_bucket_files(args.bucket, args.prefix, args.region)

        if args.output == "json":
            print(json.dumps(files, indent=2))
        else:
            print_files_table(files)


if __name__ == "__main__":
    main()
