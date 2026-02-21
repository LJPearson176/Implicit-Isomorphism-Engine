import sys
import unittest
from unittest.mock import MagicMock
import os
import tarfile
import io
import shutil

# Mock arxiv
sys.modules["arxiv"] = MagicMock()

# Now import fetch_source
import fetch_source

class TestFetchSourceSecurity(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.abspath("test_safe_fetch")
        os.makedirs(self.test_dir, exist_ok=True)
        self.tar_path = os.path.join(self.test_dir, "malicious.tar.gz")
        self.create_malicious_tar(self.tar_path)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def create_malicious_tar(self, tar_path):
        with tarfile.open(tar_path, "w:gz") as tar:
            # Create a file that traverses up
            info = tarfile.TarInfo(name="../pwned_safe.txt")
            content = b"You have been pwned!"
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))

    def test_fetch_source_malicious_tar(self):
        # Mock arxiv behavior
        mock_client = sys.modules["arxiv"].Client.return_value
        mock_result = MagicMock()
        # Make the search return our mock result
        mock_client.results.return_value = iter([mock_result])

        # When download_source is called, return our malicious tar path
        # But fetch_source expects download_source to download to a path.
        # It calls: paper.download_source(dirpath=output_dir, filename=f"{arxiv_id}.tar.gz")
        # We need to ensure the file exists there.

        def side_effect_download(dirpath, filename):
            dest_path = os.path.join(dirpath, filename)
            shutil.copy(self.tar_path, dest_path)
            return dest_path

        mock_result.download_source.side_effect = side_effect_download

        # Call fetch_source
        arxiv_id = "malicious_id"
        output_dir = os.path.join(self.test_dir, "output")

        print("Calling fetch_source with malicious tar...")
        result = fetch_source.fetch_source(arxiv_id, output_dir=output_dir)

        # Assertions
        self.assertIsNone(result, "fetch_source should return None for malicious tar")

        # Check that the file was NOT created outside
        # If extraction is to output_dir/arxiv_id, and file is ../pwned.txt,
        # it would end up in output_dir/pwned_safe.txt.

        pwned_path = os.path.join(output_dir, "pwned_safe.txt")

        self.assertFalse(os.path.exists(pwned_path), f"Malicious file should not exist at {pwned_path}")

        print("\nTest passed: Malicious tar was rejected.")

if __name__ == "__main__":
    unittest.main()
