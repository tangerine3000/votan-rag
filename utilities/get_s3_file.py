import boto3
import os

def download_file_from_s3(bucket_name, s3_key, local_file_path=None):
    """
    Download a file from S3 bucket.
    
    Args:
        bucket_name: Name of the S3 bucket
        s3_key: Key (path) of the file in S3
        local_file_path: Optional local path to save the file
    
    Returns:
        File content as string or path to downloaded file
    """
    s3_client = boto3.client('s3')
    
    try:
        if local_file_path:
            # Download to local file
            s3_client.download_file(bucket_name, s3_key, local_file_path)
            print(f"File downloaded to: {local_file_path}")
            return local_file_path
        else:
            # Get file content directly
            response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
            content = response['Body'].read()
            
            # Decode if text file
            try:
                content = content.decode('utf-8')
            except UnicodeDecodeError:
                pass  # Return bytes if not text
            
            print(f"File content retrieved from s3://{bucket_name}/{s3_key}")
            return content
            
    except Exception as e:
        print(f"Error downloading file: {e}")
        raise


def list_s3_files(bucket_name, prefix=''):
    """
    List files in an S3 bucket with optional prefix.
    
    Args:
        bucket_name: Name of the S3 bucket
        prefix: Optional prefix to filter files
    
    Returns:
        List of file keys
    """
    s3_client = boto3.client('s3')
    
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        
        if 'Contents' in response:
            files = [obj['Key'] for obj in response['Contents']]
            print(f"Found {len(files)} files in s3://{bucket_name}/{prefix}")
            return files
        else:
            print(f"No files found in s3://{bucket_name}/{prefix}")
            return []
            
    except Exception as e:
        print(f"Error listing files: {e}")
        raise


# Example usage
if __name__ == "__main__":
    # Configuration
    BUCKET_NAME = "your-bucket-name"
    FILE_KEY = "path/to/your/file.txt"
    
    # Method 1: Get content directly
    content = download_file_from_s3(BUCKET_NAME, FILE_KEY)
    print("\nFile content preview:")
    print(content[:500] if isinstance(content, str) else content[:500])
    
    # Method 2: Download to local file
    # local_path = "downloaded_file.txt"
    # download_file_from_s3(BUCKET_NAME, FILE_KEY, local_path)
    
    # Method 3: List files in bucket
    # files = list_s3_files(BUCKET_NAME, prefix="sample_assets/")
    # for file in files:
    #     print(f"  - {file}")
