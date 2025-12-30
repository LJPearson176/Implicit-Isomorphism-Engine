import os
import boto3
from botocore.exceptions import ClientError

class S3Manager:
    def __init__(self):
        self.bucket = os.environ.get("WASABI_BUCKET", "kgcn")
        self.client = boto3.client(
            's3',
            endpoint_url=os.environ.get("WASABI_ENDPOINT", "https://s3.us-east-1.wasabisys.com"),
            aws_access_key_id=os.environ.get("WASABI_ACCESS_KEY"),
            aws_secret_access_key=os.environ.get("WASABI_SECRET_KEY")
        )

    def upload_file(self, file_path, object_name=None):
        if object_name is None:
            object_name = os.path.basename(file_path)
        try:
            self.client.upload_file(file_path, self.bucket, object_name)
            print(f"Uploaded {file_path} to s3://{self.bucket}/{object_name}")
            return True
        except ClientError as e:
            print(f"Upload failed: {e}")
            return False

    def download_file(self, object_name, file_path):
        try:
            self.client.download_file(self.bucket, object_name, file_path)
            print(f"Downloaded s3://{self.bucket}/{object_name} to {file_path}")
            return True
        except ClientError as e:
            print(f"Download failed: {e}")
            return False

    def list_files(self, prefix=""):
        try:
            response = self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            return []
        except ClientError as e:
            print(f"List failed: {e}")
            return []

    def move_file(self, source_key, dest_key):
        copy_source = {'Bucket': self.bucket, 'Key': source_key}
        try:
            self.client.copy_object(CopySource=copy_source, Bucket=self.bucket, Key=dest_key)
            self.client.delete_object(Bucket=self.bucket, Key=source_key)
            print(f"Moved {source_key} -> {dest_key}")
            return True
        except ClientError as e:
            print(f"Move failed: {e}")
            return False
