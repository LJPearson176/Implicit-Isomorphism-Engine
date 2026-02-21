import sys
import unittest
from unittest.mock import MagicMock, patch
import os
import importlib

# Ensure verifier is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

class MockFigure: pass

class TestVerifier(unittest.TestCase):
    def setUp(self):
        # Create patches for modules
        self.modules_patcher = patch.dict(sys.modules, {
            "numpy": MagicMock(),
            "matplotlib": MagicMock(),
            "matplotlib.pyplot": MagicMock(),
            "torch": MagicMock(),
        })
        self.modules_patcher.start()

        # Configure mocks specifically
        mock_plt = sys.modules["matplotlib.pyplot"]
        mock_plt.Figure = MockFigure
        # Ensure matplotlib.pyplot attribute on matplotlib mock points to our mock_plt
        sys.modules["matplotlib"].pyplot = mock_plt

        # Import verifier now that dependencies are mocked
        # Since we use patch.dict on sys.modules, verifier will be removed after tearDown
        # if it wasn't present before. So each test imports a fresh verifier module.
        import verifier
        # importlib.reload(verifier) # Not strictly necessary if patch.dict works as expected, but safe.
        self.verifier = verifier

        self.created_files = []
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        importlib.invalidate_caches()

    def tearDown(self):
        self.modules_patcher.stop()
        for f in self.created_files:
            if os.path.exists(f):
                os.remove(f)

    def create_dummy_file(self, filename, content):
        with open(filename, "w") as f:
            f.write(content)
        self.created_files.append(filename)
        importlib.invalidate_caches()

    def test_simulate_opinion_valid(self):
        code = """
import matplotlib.pyplot as plt
def simulate_opinion():
    print("Simulating opinion...")
    return plt.figure()
"""
        filename = "temp_valid_opinion.py"
        module_name = "temp_valid_opinion"
        self.create_dummy_file(filename, code)

        result = self.verifier.verify_generated_code(module_name)
        self.assertTrue(result["success"])
        self.assertIn("Successfully executed simulate_opinion", result["message"])

    def test_custom_physics_loss_valid(self):
        code = """
import torch
def custom_physics_loss(x, y, z):
    return torch.tensor(0.0)
"""
        filename = "temp_valid_loss.py"
        module_name = "temp_valid_loss"
        self.create_dummy_file(filename, code)

        result = self.verifier.verify_generated_code(module_name)
        self.assertTrue(result["success"])
        self.assertIn("Successfully executed custom_physics_loss", result["message"])

    def test_simulate_quantum_dynamics_valid(self):
        code = """
def simulate_quantum_dynamics():
    return "Quantum result"
"""
        filename = "temp_valid_quantum.py"
        module_name = "temp_valid_quantum"
        self.create_dummy_file(filename, code)

        result = self.verifier.verify_generated_code(module_name)
        self.assertTrue(result["success"])
        self.assertIn("Successfully executed simulate_quantum_dynamics", result["message"])

    def test_syntax_error(self):
        code = """
def invalid_syntax(:
    pass
"""
        filename = "temp_syntax_error.py"
        module_name = "temp_syntax_error"
        self.create_dummy_file(filename, code)

        result = self.verifier.verify_generated_code(module_name)
        self.assertFalse(result["success"])
        self.assertIn("Import/Syntax Error", result["error"])

    def test_runtime_error(self):
        code = """
def simulate_opinion():
    raise ValueError("Something went wrong")
"""
        filename = "temp_runtime_error.py"
        module_name = "temp_runtime_error"
        self.create_dummy_file(filename, code)

        result = self.verifier.verify_generated_code(module_name)
        self.assertFalse(result["success"])
        self.assertIn("Runtime Error", result["error"])

    def test_no_valid_function(self):
        code = """
def some_other_function():
    pass
"""
        filename = "temp_no_func.py"
        module_name = "temp_no_func"
        self.create_dummy_file(filename, code)

        result = self.verifier.verify_generated_code(module_name)
        self.assertFalse(result["success"])
        self.assertIn("No standard simulation function found", result["error"])

if __name__ == "__main__":
    unittest.main()
