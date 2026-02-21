import unittest
from unittest.mock import patch, MagicMock
from lab_interface import LabInterface
import os

class TestLabInterface(unittest.TestCase):
    @patch('lab_interface.os.path.exists')
    @patch('lab_interface.os.makedirs')
    def test_init_creates_dir(self, mock_makedirs, mock_exists):
        # Simulate directory does not exist
        mock_exists.return_value = False
        LabInterface()
        # Verify makedirs is called with the correct directory
        mock_makedirs.assert_called_with("lab_deployment")

    @patch('lab_interface.os.path.exists')
    @patch('lab_interface.os.makedirs')
    def test_init_skips_dir_if_exists(self, mock_makedirs, mock_exists):
        # Simulate directory already exists
        mock_exists.return_value = True
        LabInterface()
        # Verify makedirs is NOT called
        mock_makedirs.assert_not_called()

    @patch('lab_interface.os.path.exists')
    @patch('lab_interface.os.makedirs') # Mock makedirs just in case to prevent actual dir creation
    def test_submit_experiment_missing_source(self, mock_makedirs, mock_exists):
        # First call is for deployment_dir (in init), second for source file
        mock_exists.side_effect = [True, False]
        lab = LabInterface()
        result = lab.submit_experiment("missing_module")
        self.assertEqual(result, {"success": False, "error": "Source file not found."})

    @patch('lab_interface.time.time')
    @patch('lab_interface.time.sleep')
    @patch('lab_interface.shutil.copy')
    @patch('lab_interface.os.path.exists')
    @patch('lab_interface.os.makedirs') # Mock makedirs
    def test_submit_experiment_success(self, mock_makedirs, mock_exists, mock_copy, mock_sleep, mock_time):
        # timestamps for filename and job_id
        mock_time.return_value = 1000
        # First call is for deployment_dir (init), second for source file
        mock_exists.side_effect = [True, True]

        lab = LabInterface()
        result = lab.submit_experiment("test_module")

        expected_target = os.path.join("lab_deployment", "test_module_v1000.py")
        mock_copy.assert_called_with("test_module.py", expected_target)

        self.assertEqual(result["success"], True)
        self.assertEqual(result["job_id"], "job-1000")
        self.assertEqual(result["status"], "QUEUED")

if __name__ == '__main__':
    unittest.main()
