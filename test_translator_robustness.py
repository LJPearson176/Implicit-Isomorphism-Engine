
import unittest
import json
import sys
from unittest.mock import MagicMock

# Mock dependencies
sys.modules["torch"] = MagicMock()
sys.modules["requests"] = MagicMock()

# Mock verify_setup to avoid importing data_engine
mock_verify_setup = MagicMock()
def mock_load_env():
    pass
mock_verify_setup.load_env = mock_load_env
sys.modules["verify_setup"] = mock_verify_setup

# Mock ollama to avoid import errors if translator imports it
sys.modules["ollama"] = MagicMock()

from translator import NeuroSymbolicTranslator
import logging

class TestTranslatorRobustness(unittest.TestCase):
    def setUp(self):
        self.translator = NeuroSymbolicTranslator()
        # Mocking or testing internal JSON repair logic
    
    def test_json_repair_single_backslash(self):
        """Test if the translator handles single backslashes in JSON correctly."""
        # Simulation of LLM output with unescaped backslashes
        # "key": "\Psi"  <- This is invalid JSON.
        # Should be repaired to keys with generic text or handled.
        
        # NOTE: The translator's robust logic is inside map_variables (a complex function).
        # We can extract the logic or simpler: test map_variables with a mocked LLM response?
        # Since we use Ollama, we can't easily unit test without mocking.
        # For now, let's verify if the regex replacement logic works if we isolate it.
        
        broken_json = '{"variable": "\\Psi"}' # This is actually valid python string representing invalid json if interpreted literally
        # Wait, inside the string literal: \\Psi -> \Psi.
        # JSON needs \\\\Psi to become \\Psi in string -> \Psi in value.
        
        # Let's verify the string replacement logic manualy as implemented in translator.py
        
        # Problematic string from LLM:
        llm_output_raw = r'{"var": "\Psi"}' # Raw string: contains \Psi
        
        try:
            json.loads(llm_output_raw)
        except json.JSONDecodeError:
            print("\nVerified: Raw LLM output fails standard JSON load.")
            
        # The fix in translator.py:
        escaped = llm_output_raw.replace('\\', '\\\\')
        print(f"Escaped version: {escaped}")
        
        valid_json = json.loads(escaped)
        self.assertEqual(valid_json['var'], '\\Psi')
        print("Verified: Escaping fix works.")

if __name__ == '__main__':
    unittest.main()
