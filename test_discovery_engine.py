import sys
import unittest
import tempfile
import shutil
import os
from unittest.mock import MagicMock

# Recursive mock for submodules
class MockModule(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

# Mock external dependencies
# We need to mock top-level packages and their submodules that are imported directly
sys.modules["torch"] = MagicMock()
sys.modules["torch.nn"] = MagicMock()
sys.modules["torch.nn.functional"] = MagicMock()
sys.modules["torch_geometric"] = MagicMock()
sys.modules["torch_geometric.nn"] = MagicMock()
sys.modules["torch_geometric.data"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["supabase"] = MagicMock()
sys.modules["arxiv"] = MagicMock()
sys.modules["googleapiclient"] = MagicMock()
sys.modules["googleapiclient.discovery"] = MagicMock()
sys.modules["requests"] = MagicMock()
sys.modules["ollama"] = MagicMock()

# Import the module to test
# We wrap this in a try-except block just in case other imports fail,
# but for the test purpose, if this fails, the test fails.
try:
    from discovery_engine import find_tex_files
except ImportError as e:
    # If mocking failed, we might see an error here.
    # But since we are mocking modules in sys.modules, standard imports should work.
    raise e

class TestFindTexFiles(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def create_file(self, filename):
        path = os.path.join(self.test_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write("content")
        return path

    def test_find_single_tex_file(self):
        tex_file = self.create_file("paper.tex")
        found = find_tex_files(self.test_dir)
        self.assertEqual(len(found), 1)
        self.assertEqual(os.path.abspath(found[0]), os.path.abspath(tex_file))

    def test_find_multiple_tex_files(self):
        f1 = self.create_file("paper1.tex")
        f2 = self.create_file("paper2.tex")
        found = find_tex_files(self.test_dir)
        self.assertEqual(len(found), 2)
        found_abs = sorted([os.path.abspath(f) for f in found])
        expected_abs = sorted([os.path.abspath(f1), os.path.abspath(f2)])
        self.assertEqual(found_abs, expected_abs)

    def test_nested_directories(self):
        f1 = self.create_file("root.tex")
        f2 = self.create_file("subdir/nested.tex")
        found = find_tex_files(self.test_dir)
        self.assertEqual(len(found), 2)
        found_abs = sorted([os.path.abspath(f) for f in found])
        expected_abs = sorted([os.path.abspath(f1), os.path.abspath(f2)])
        self.assertEqual(found_abs, expected_abs)

    def test_ignore_non_tex_files(self):
        self.create_file("image.png")
        self.create_file("readme.txt")
        tex = self.create_file("main.tex")
        found = find_tex_files(self.test_dir)
        self.assertEqual(len(found), 1)
        self.assertEqual(os.path.abspath(found[0]), os.path.abspath(tex))

    def test_empty_directory(self):
        found = find_tex_files(self.test_dir)
        self.assertEqual(len(found), 0)

if __name__ == '__main__':
    unittest.main()
