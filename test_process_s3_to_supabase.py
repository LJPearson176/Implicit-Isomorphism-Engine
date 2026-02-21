import unittest
import sys
import os
from unittest.mock import MagicMock, patch, mock_open

# Mock dependencies that might be missing in the test environment
# We must do this BEFORE importing the modules that use them
sys.modules["boto3"] = MagicMock()
sys.modules["botocore"] = MagicMock()
sys.modules["botocore.exceptions"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["supabase"] = MagicMock()
sys.modules["torch_geometric"] = MagicMock()
sys.modules["torch_geometric.data"] = MagicMock()

# Mock verify_setup to avoid side effects during import
sys.modules["verify_setup"] = MagicMock()

# Now import the module under test
from process_s3_to_supabase import process_s3_to_supabase

class TestProcessS3ToSupabase(unittest.TestCase):

    @patch("process_s3_to_supabase.S3Manager")
    @patch("process_s3_to_supabase.DataEngine")
    @patch("process_s3_to_supabase.os.path.exists")
    @patch("process_s3_to_supabase.os.remove")
    @patch("builtins.open", new_callable=mock_open, read_data='{"arxiv_id": "1", "title": "Test Paper", "abstract": "Abstract"}')
    def test_happy_path(self, mock_file, mock_remove, mock_exists, MockDataEngine, MockS3Manager):
        # Setup mocks
        mock_s3 = MockS3Manager.return_value
        mock_engine = MockDataEngine.return_value

        mock_s3.list_files.return_value = ["raw/test.jsonl"]
        mock_s3.download_file.return_value = True
        # cleanup check
        mock_exists.return_value = True

        # Run
        process_s3_to_supabase()

        # Verify list and download
        mock_s3.list_files.assert_called_with(prefix="raw/")
        mock_s3.download_file.assert_called_with("raw/test.jsonl", "temp_test.jsonl")

        # Check if ingest was called with correct data
        mock_engine.ingest_arxiv_batch.assert_called_once()
        args, _ = mock_engine.ingest_arxiv_batch.call_args
        self.assertEqual(len(args[0]), 1)
        self.assertEqual(args[0][0]["arxiv_id"], "1")

        # Check move
        mock_s3.move_file.assert_called_with("raw/test.jsonl", "processed/test.jsonl")

        # Check cleanup
        mock_remove.assert_called_with("temp_test.jsonl")

    @patch("process_s3_to_supabase.S3Manager")
    @patch("process_s3_to_supabase.DataEngine")
    def test_no_files(self, MockDataEngine, MockS3Manager):
        mock_s3 = MockS3Manager.return_value
        mock_s3.list_files.return_value = []

        process_s3_to_supabase()

        mock_s3.download_file.assert_not_called()

    @patch("process_s3_to_supabase.S3Manager")
    @patch("process_s3_to_supabase.DataEngine")
    def test_ignore_non_jsonl(self, MockDataEngine, MockS3Manager):
        mock_s3 = MockS3Manager.return_value
        mock_s3.list_files.return_value = ["raw/image.png"]

        process_s3_to_supabase()

        mock_s3.download_file.assert_not_called()

    @patch("process_s3_to_supabase.S3Manager")
    @patch("process_s3_to_supabase.DataEngine")
    @patch("process_s3_to_supabase.os.path.exists")
    @patch("process_s3_to_supabase.os.remove")
    def test_download_failure(self, mock_remove, mock_exists, MockDataEngine, MockS3Manager):
        mock_s3 = MockS3Manager.return_value
        mock_s3.list_files.return_value = ["raw/test.jsonl"]
        mock_s3.download_file.return_value = False # Simulate failure

        process_s3_to_supabase()

        # Should attempt download
        mock_s3.download_file.assert_called_with("raw/test.jsonl", "temp_test.jsonl")

        # Should not move or ingest
        mock_s3.move_file.assert_not_called()
        MockDataEngine.return_value.ingest_arxiv_batch.assert_not_called()

        # Cleanup should not happen because of 'continue'
        mock_remove.assert_not_called()

    @patch("process_s3_to_supabase.S3Manager")
    @patch("process_s3_to_supabase.DataEngine")
    @patch("process_s3_to_supabase.os.path.exists")
    @patch("process_s3_to_supabase.os.remove")
    @patch("builtins.open", new_callable=mock_open, read_data='{"arxiv_id": "1", "title": "Test Paper", "abstract": "Abstract"}')
    def test_ingestion_failure(self, mock_file, mock_remove, mock_exists, MockDataEngine, MockS3Manager):
        mock_s3 = MockS3Manager.return_value
        mock_engine = MockDataEngine.return_value

        mock_s3.list_files.return_value = ["raw/test.jsonl"]
        mock_s3.download_file.return_value = True
        mock_exists.return_value = True

        # Simulate ingestion error
        mock_engine.ingest_arxiv_batch.side_effect = Exception("DB Error")

        process_s3_to_supabase()

        # Verify ingest called
        mock_engine.ingest_arxiv_batch.assert_called_once()

        # Verify move NOT called
        mock_s3.move_file.assert_not_called()

        # Verify cleanup called (it happens after try/except block)
        mock_remove.assert_called_with("temp_test.jsonl")

if __name__ == "__main__":
    unittest.main()
