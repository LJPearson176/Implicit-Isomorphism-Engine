import sys
from unittest.mock import MagicMock

# Mock external dependencies that might be missing in the environment
# These must be mocked BEFORE importing the modules that use them
sys.modules["torch"] = MagicMock()
sys.modules["torch.nn"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["scipy"] = MagicMock()
sys.modules["scipy.integrate"] = MagicMock()
sys.modules["scipy.linalg"] = MagicMock()
sys.modules["networkx"] = MagicMock()
sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()
sys.modules["arxiv"] = MagicMock()
sys.modules["requests"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["supabase"] = MagicMock()
sys.modules["torch_geometric"] = MagicMock()
sys.modules["torch_geometric.data"] = MagicMock()

import unittest
from unittest.mock import patch, mock_open
from compiler import Compiler

class TestCompiler(unittest.TestCase):
    @patch('compiler.NeuroSymbolicTranslator')
    def setUp(self, MockTranslator):
        self.mock_translator_class = MockTranslator
        self.compiler = Compiler()
        # The translator instance inside compiler is the return value of the mocked class
        self.mock_translator_instance = self.compiler.translator

    def test_compile_success(self):
        """Test successful compilation writing to file."""
        code_string = 'print("Hello World")'
        expected_filename = "generated_loss.py"

        # Base imports defined in Compiler.compile
        base_imports = ["import torch", "import torch.nn as nn", "import numpy as np"]
        expected_content = "\n".join(base_imports + [code_string])

        with patch("builtins.open", mock_open()) as mocked_file:
            result = self.compiler.compile(code_string)

            self.assertTrue(result)
            mocked_file.assert_called_with(expected_filename, "w")
            mocked_file().write.assert_called_with(expected_content)

    def test_compile_duplicate_imports(self):
        """Test that duplicate base imports are skipped."""
        # Code with duplicate import
        code_string = 'import torch\nprint("Hello")'

        # expected content should not have duplicate import torch
        base_imports = ["import torch", "import torch.nn as nn", "import numpy as np"]
        # 'import torch' matches base import, so it is skipped
        expected_content = "\n".join(base_imports + ['print("Hello")'])

        with patch("builtins.open", mock_open()) as mocked_file:
            result = self.compiler.compile(code_string)

            self.assertTrue(result)
            mocked_file().write.assert_called_with(expected_content)

    def test_compile_failure(self):
        """Test compilation failure when file write fails."""
        code_string = 'print("Fail")'

        with patch("builtins.open", mock_open()) as mocked_file:
            # Configure mock to raise IOError on write or open
            mocked_file.side_effect = IOError("Disk full")

            result = self.compiler.compile(code_string)
            self.assertFalse(result)

    def test_compile_isomorphism_success(self):
        """Test compile_isomorphism successful flow."""
        bridge_edge = {
            'source_equation_name': 'Navier-Stokes',
            'variable_mapping': {'u': 'price'},
            'target_domain': 'Financial Markets'
        }

        generated_code = 'print("Generated Code")'
        self.mock_translator_instance.generate_code.return_value = generated_code

        # Mock compile method to isolate verify it calls compile
        with patch.object(self.compiler, 'compile', return_value=True) as mock_compile:
            result = self.compiler.compile_isomorphism(bridge_edge)

            self.assertEqual(result, "generated_scout_solution.py")
            # Verify translator called correctly
            eq_context = "Navier-Stokes (Mapping: {'u': 'price'})"
            self.mock_translator_instance.generate_code.assert_called_with(eq_context, target_domain='Financial Markets')
            # Verify compile called with generated code
            mock_compile.assert_called_with(generated_code)
            # Verify output file changed
            self.assertEqual(self.compiler.output_file, "generated_scout_solution.py")

    def test_compile_isomorphism_failure_no_code(self):
        """Test compile_isomorphism when no code is generated."""
        bridge_edge = {
            'source_equation_name': 'Navier-Stokes',
            'variable_mapping': {'u': 'price'},
            'target_domain': 'Financial Markets'
        }

        self.mock_translator_instance.generate_code.return_value = None

        result = self.compiler.compile_isomorphism(bridge_edge)
        self.assertIsNone(result)

    def test_compile_isomorphism_failure_compile_false(self):
        """Test compile_isomorphism when compilation fails."""
        bridge_edge = {
            'source_equation_name': 'Navier-Stokes',
            'variable_mapping': {'u': 'price'},
            'target_domain': 'Financial Markets'
        }

        generated_code = 'print("Generated Code")'
        self.mock_translator_instance.generate_code.return_value = generated_code

        with patch.object(self.compiler, 'compile', return_value=False):
            result = self.compiler.compile_isomorphism(bridge_edge)
            self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
