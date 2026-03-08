import argparse
import os
import boto3


def upload_file_to_s3(bucket_name, local_file_path, s3_key=None):
    """
    Upload a single file to an S3 bucket.

    Args:
        bucket_name: Name of the S3 bucket
        local_file_path: Path to local file
        s3_key: Optional key/path in S3 (defaults to file name)

    Returns:
        S3 URI of uploaded file
    """
    s3_client = boto3.client('s3')

    if not os.path.isfile(local_file_path):
        raise FileNotFoundError(f"Local file not found: {local_file_path}")

    if not s3_key:
        s3_key = os.path.basename(local_file_path)

    try:
        s3_client.upload_file(local_file_path, bucket_name, s3_key)
        s3_uri = f"s3://{bucket_name}/{s3_key}"
        print(f"Uploaded: {local_file_path} -> {s3_uri}")
        return s3_uri
    except Exception as e:
        print(f"Error uploading file: {e}")
        raise


def upload_folder_to_s3(bucket_name, local_folder_path, s3_prefix=""):
    """
    Upload all files from a local folder to an S3 bucket.

    Args:
        bucket_name: Name of the S3 bucket
        local_folder_path: Path to local folder
        s3_prefix: Optional prefix/folder in S3

    Returns:
        Number of uploaded files
    """
    if not os.path.isdir(local_folder_path):
        raise NotADirectoryError(f"Local folder not found: {local_folder_path}")

    uploaded_count = 0

    for root, _, files in os.walk(local_folder_path):
        for file_name in files:
            local_file_path = os.path.join(root, file_name)
            relative_path = os.path.relpath(local_file_path, local_folder_path).replace('\\', '/')

            if s3_prefix:
                s3_key = f"{s3_prefix.rstrip('/')}/{relative_path}"
            else:
                s3_key = relative_path

            upload_file_to_s3(bucket_name, local_file_path, s3_key)
            uploaded_count += 1

    print(f"Total uploaded files: {uploaded_count}")
    return uploaded_count


def main():
    parser = argparse.ArgumentParser(description="Upload files to an S3 bucket")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--file", help="Path to a local file to upload")
    parser.add_argument("--folder", help="Path to a local folder to upload")
    parser.add_argument("--key", help="S3 key for single file upload")
    parser.add_argument("--prefix", default="", help="S3 prefix for folder upload")

    args = parser.parse_args()

    if args.file and args.folder:
        raise ValueError("Use either --file or --folder, not both.")

    if not args.file and not args.folder:
        raise ValueError("You must provide either --file or --folder.")

    if args.file:
        upload_file_to_s3(args.bucket, args.file, args.key)
    else:
        upload_folder_to_s3(args.bucket, args.folder, args.prefix)


if __name__ == "__main__":
    main()
