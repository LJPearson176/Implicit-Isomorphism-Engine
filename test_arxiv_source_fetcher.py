
import sys
from unittest.mock import MagicMock

# Mock dependencies that are missing in the environment
sys.modules['torch'] = MagicMock()
sys.modules['requests'] = MagicMock()
sys.modules['guardrail'] = MagicMock()
sys.modules['verify_setup'] = MagicMock()
# guardrail and verify_setup are local files, they might exist, but if they import torch/requests, they fail.
# Let's check if they exist. They do. But safely mocking them avoids transitive import errors.

# Also mock ollama if it's not there (it is imported locally in methods, but good practice)
sys.modules['ollama'] = MagicMock()

import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import tarfile

# Now import the class under test
# We need to make sure we don't fail if translator imports other things
try:
    from translator import ArxivSourceFetcher
except ImportError:
    # If explicit import fails, we might need to mock more
    pass

class TestArxivSourceFetcher(unittest.TestCase):

    @patch('translator.os.makedirs')
    @patch('translator.os.path.exists')
    def setUp(self, mock_exists, mock_makedirs):
        # Prevent actual directory creation during init
        mock_exists.return_value = False
        self.fetcher = ArxivSourceFetcher()
        self.fetcher.download_dir = "/mock/cache/dir"

    @patch('translator.os.walk')
    @patch('translator.os.path.exists')
    def test_fetch_source_cached(self, mock_exists, mock_walk):
        # Setup
        arxiv_id = "1234.5678"
        cache_path = os.path.join(self.fetcher.download_dir, arxiv_id)

        # Mock that cache exists
        def exists_side_effect(path):
            if path == self.fetcher.download_dir:
                return True
            if path == cache_path:
                return True
            return False
        mock_exists.side_effect = exists_side_effect

        # Mock finding files
        mock_walk.return_value = [
            (cache_path, [], ["main.tex", "other.txt"])
        ]

        # Execute
        result = self.fetcher.fetch_source(arxiv_id)

        # Verify
        expected_file = os.path.join(cache_path, "main.tex")
        self.assertIn(expected_file, result)
        mock_walk.assert_called_once()

    @patch('translator.os.walk')
    @patch('translator.tarfile.open')
    @patch('translator.requests.get')
    @patch('translator.os.makedirs')
    @patch('translator.os.path.exists')
    def test_fetch_source_download_success_tar(self, mock_exists, mock_makedirs, mock_get, mock_tarfile, mock_walk):
        # Setup
        arxiv_id = "1234.5678"
        cache_path = os.path.join(self.fetcher.download_dir, arxiv_id)

        # Mock that cache does NOT exist initially
        mock_exists.return_value = False

        # Mock successful download
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake_tar_content"
        mock_get.return_value = mock_response

        # Mock tarfile extraction
        mock_tar = MagicMock()
        mock_tarfile.return_value.__enter__.return_value = mock_tar

        # Mock finding files after extraction
        mock_walk.return_value = [
            (cache_path, [], ["paper.tex"])
        ]

        # Execute
        result = self.fetcher.fetch_source(arxiv_id)

        # Verify
        mock_get.assert_called_with(f"{self.fetcher.base_url}{arxiv_id}", stream=True)
        # Verify os.makedirs was called for cache_path
        mock_makedirs.assert_any_call(cache_path, exist_ok=True)
        mock_tar.extractall.assert_called_with(path=cache_path)
        self.assertEqual(result, [os.path.join(cache_path, "paper.tex")])

    @patch('translator.os.walk')
    @patch('builtins.open', new_callable=mock_open)
    @patch('translator.tarfile.open')
    @patch('translator.requests.get')
    @patch('translator.os.makedirs')
    @patch('translator.os.path.exists')
    def test_fetch_source_download_success_single_file(self, mock_exists, mock_makedirs, mock_get, mock_tarfile, mock_file, mock_walk):
        # Setup
        arxiv_id = "1234.5678"
        cache_path = os.path.join(self.fetcher.download_dir, arxiv_id)

        # Mock that cache does NOT exist
        mock_exists.return_value = False

        # Mock successful download
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake_tex_content"
        mock_get.return_value = mock_response

        # Mock tarfile failure
        mock_tarfile.side_effect = tarfile.ReadError("Not a tar file")

        # Mock finding files after saving single file
        mock_walk.return_value = [
            (cache_path, [], ["main.tex"])
        ]

        # Execute
        result = self.fetcher.fetch_source(arxiv_id)

        # Verify
        mock_tarfile.assert_called_once() # Should have tried to open as tar
        mock_file.assert_called_with(os.path.join(cache_path, "main.tex"), "wb")
        mock_file().write.assert_called_with(b"fake_tex_content")
        self.assertEqual(result, [os.path.join(cache_path, "main.tex")])

    @patch('translator.requests.get')
    @patch('translator.os.path.exists')
    def test_fetch_source_download_failure(self, mock_exists, mock_get):
        # Setup
        arxiv_id = "1234.5678"
        mock_exists.return_value = False

        # Mock failed download
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        # Execute
        result = self.fetcher.fetch_source(arxiv_id)

        # Verify
        self.assertEqual(result, [])

    @patch('translator.requests.get')
    @patch('translator.os.path.exists')
    def test_fetch_source_exception(self, mock_exists, mock_get):
        # Setup
        arxiv_id = "1234.5678"
        mock_exists.return_value = False

        # Mock exception
        # We need to make sure we are using the mocked requests exception if requests is mocked
        import requests
        # If requests is a MagicMock, RequestException is also a MagicMock (or we can set it)
        # But translator code catches Exception, so any exception works.
        mock_get.side_effect = Exception("Network error")

        # Execute
        result = self.fetcher.fetch_source(arxiv_id)

        # Verify
        self.assertEqual(result, [])

if __name__ == '__main__':
    unittest.main()
