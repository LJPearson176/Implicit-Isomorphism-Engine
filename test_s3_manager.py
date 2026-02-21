import unittest
from unittest.mock import MagicMock
import sys
import os

# --- Mocking dependencies BEFORE import ---
# Since boto3/botocore might not be installed, we mock them at sys.modules level.

# Define a mock ClientError that inherits from Exception so it can be caught
class MockClientError(Exception):
    def __init__(self, error_response, operation_name):
        self.response = error_response
        self.operation_name = operation_name
        super().__init__(f"An error occurred ({error_response['Error']['Code']}) when calling the {operation_name} operation: {error_response['Error']['Message']}")

mock_boto3 = MagicMock()
mock_botocore = MagicMock()
mock_botocore_exceptions = MagicMock()
mock_botocore_exceptions.ClientError = MockClientError

sys.modules['boto3'] = mock_boto3
sys.modules['botocore'] = mock_botocore
sys.modules['botocore.exceptions'] = mock_botocore_exceptions

# Ensure the module can be imported
sys.path.append(os.getcwd())

# Now we can import s3_manager
from s3_manager import S3Manager

class TestS3Manager(unittest.TestCase):
    def setUp(self):
        # Reset the mock client before each test
        mock_boto3.client.reset_mock()
        self.mock_s3 = MagicMock()
        mock_boto3.client.return_value = self.mock_s3

        # Initialize manager which will use the mocked client from sys.modules
        self.manager = S3Manager()

    def test_init(self):
        # Verify that boto3.client was called with correct default args
        mock_boto3.client.assert_called_with(
            's3',
            endpoint_url="https://s3.us-east-1.wasabisys.com",
            aws_access_key_id=None,
            aws_secret_access_key=None
        )
        self.assertEqual(self.manager.bucket, "kgcn")

    def test_upload_file_success(self):
        file_path = "test.txt"
        result = self.manager.upload_file(file_path)
        self.assertTrue(result)
        self.mock_s3.upload_file.assert_called_with(file_path, "kgcn", "test.txt")

    def test_upload_file_custom_name(self):
        file_path = "test.txt"
        object_name = "custom.txt"
        result = self.manager.upload_file(file_path, object_name)
        self.assertTrue(result)
        self.mock_s3.upload_file.assert_called_with(file_path, "kgcn", object_name)

    def test_upload_file_failure(self):
        # Simulate ClientError
        self.mock_s3.upload_file.side_effect = MockClientError({'Error': {'Code': '500', 'Message': 'Error'}}, 'upload_file')
        result = self.manager.upload_file("test.txt")
        self.assertFalse(result)

    def test_download_file_success(self):
        object_name = "test.txt"
        file_path = "downloaded.txt"
        result = self.manager.download_file(object_name, file_path)
        self.assertTrue(result)
        self.mock_s3.download_file.assert_called_with("kgcn", object_name, file_path)

    def test_download_file_failure(self):
        self.mock_s3.download_file.side_effect = MockClientError({'Error': {'Code': '404', 'Message': 'Not Found'}}, 'download_file')
        result = self.manager.download_file("test.txt", "downloaded.txt")
        self.assertFalse(result)

    def test_list_files_success(self):
        self.mock_s3.list_objects_v2.return_value = {
            'Contents': [{'Key': 'file1.txt'}, {'Key': 'file2.txt'}]
        }
        result = self.manager.list_files()
        self.assertEqual(result, ['file1.txt', 'file2.txt'])
        self.mock_s3.list_objects_v2.assert_called_with(Bucket="kgcn", Prefix="")

    def test_list_files_empty(self):
        self.mock_s3.list_objects_v2.return_value = {}
        result = self.manager.list_files()
        self.assertEqual(result, [])

    def test_list_files_failure(self):
        self.mock_s3.list_objects_v2.side_effect = MockClientError({'Error': {'Code': '500', 'Message': 'Error'}}, 'list_objects_v2')
        result = self.manager.list_files()
        self.assertEqual(result, [])

    def test_move_file_success(self):
        result = self.manager.move_file("source.txt", "dest.txt")
        self.assertTrue(result)
        self.mock_s3.copy_object.assert_called_with(
            CopySource={'Bucket': 'kgcn', 'Key': 'source.txt'},
            Bucket='kgcn',
            Key='dest.txt'
        )
        self.mock_s3.delete_object.assert_called_with(Bucket='kgcn', Key='source.txt')

    def test_move_file_copy_failure(self):
        self.mock_s3.copy_object.side_effect = MockClientError({'Error': {'Code': '500', 'Message': 'Error'}}, 'copy_object')
        result = self.manager.move_file("source.txt", "dest.txt")
        self.assertFalse(result)
        self.mock_s3.delete_object.assert_not_called()

    def test_move_file_delete_failure(self):
        self.mock_s3.delete_object.side_effect = MockClientError({'Error': {'Code': '500', 'Message': 'Error'}}, 'delete_object')
        result = self.manager.move_file("source.txt", "dest.txt")
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()
