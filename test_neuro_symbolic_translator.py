import unittest
from unittest.mock import MagicMock, patch
import sys
import json
import re

# Patch ollama globally for the test execution
# This is necessary because translator.py imports ollama inside functions,
# but those imports look up sys.modules.
mock_ollama = MagicMock()
sys.modules["ollama"] = mock_ollama

# Mock other dependencies that might be missing in the test environment
sys.modules["torch"] = MagicMock()
sys.modules["requests"] = MagicMock()

# Mock verify_setup to avoid importing data_engine which triggers heavy dependencies (sentence_transformers, etc.)
# and might exit the process if they are missing.
mock_verify_setup = MagicMock()
def mock_load_env():
    pass
mock_verify_setup.load_env = mock_load_env
sys.modules["verify_setup"] = mock_verify_setup

from translator import NeuroSymbolicTranslator

class TestNeuroSymbolicTranslatorRewrite(unittest.TestCase):
    def setUp(self):
        self.translator = NeuroSymbolicTranslator()

    def test_rewrite_simple_variable(self):
        """Test simple variable replacement."""
        eq = "x + y = z"
        mapping = {"x": "Price", "y": "Volume", "z": "MarketCap"}
        expected = "Price + Volume = MarketCap"
        self.assertEqual(self.translator.rewrite_equation(eq, mapping), expected)

    def test_rewrite_latex_command(self):
        """Test replacement of LaTeX commands."""
        eq = r"\alpha + \beta"
        mapping = {r"\alpha": "GrowthRate", r"\beta": "DecayRate"}
        expected = "GrowthRate + DecayRate"
        self.assertEqual(self.translator.rewrite_equation(eq, mapping), expected)

    def test_rewrite_overlapping_variables(self):
        """Test that longer variables are replaced first."""
        eq = "x_i + x"
        mapping = {"x_i": "IndividualPrice", "x": "AveragePrice"}
        # Expect longer match to be replaced first
        expected = "IndividualPrice + AveragePrice"
        self.assertEqual(self.translator.rewrite_equation(eq, mapping), expected)

    def test_rewrite_partial_match_prevention(self):
        """Test that replacement respects word boundaries (where applicable)."""
        eq = "max(x)"
        mapping = {"x": "Price"}
        # Should NOT replace 'x' inside 'max'
        expected = "max(Price)"
        self.assertEqual(self.translator.rewrite_equation(eq, mapping), expected)

    def test_rewrite_latex_with_braces(self):
        """Test replacement inside LaTeX braces."""
        eq = r"\frac{x}{y}"
        mapping = {"x": "Numerator", "y": "Denominator"}
        expected = r"\frac{Numerator}{Denominator}"
        self.assertEqual(self.translator.rewrite_equation(eq, mapping), expected)

    def test_rewrite_complex_symbol(self):
        """Test replacement of complex LaTeX symbols."""
        eq = r"\mathbf{x}"
        mapping = {r"\mathbf{x}": "VectorPrice"}
        expected = "VectorPrice"
        self.assertEqual(self.translator.rewrite_equation(eq, mapping), expected)

    def test_rewrite_no_mapping(self):
        """Test handling of empty mapping."""
        eq = "x + y"
        mapping = {}
        expected = "x + y"
        self.assertEqual(self.translator.rewrite_equation(eq, mapping), expected)

    def test_rewrite_unmapped_variable(self):
        """Test that unmapped variables are left alone."""
        eq = "x + y"
        mapping = {"x": "Price"}
        expected = "Price + y"
        self.assertEqual(self.translator.rewrite_equation(eq, mapping), expected)

    def test_rewrite_latex_command_boundary(self):
        """Test handling of LaTeX commands which might lack clear word boundaries."""
        # Case where \b fails for LaTeX commands starting with \
        # The current implementation falls back to string replacement
        eq = r"\alpha_bet"
        mapping = {r"\alpha": "Growth"}
        # Since replace is used as fallback, it will replace \alpha with Growth
        expected = "Growth_bet"
        self.assertEqual(self.translator.rewrite_equation(eq, mapping), expected)

class TestNeuroSymbolicTranslatorMapVariables(unittest.TestCase):
    def setUp(self):
        self.translator = NeuroSymbolicTranslator()
        # Ensure sys.modules["ollama"] is OUR mock (in case other tests overwrote it)
        sys.modules["ollama"] = mock_ollama
        # Reset the mock before each test
        mock_ollama.reset_mock()

    def test_map_variables_success(self):
        """Test successful variable mapping with mocked LLM response."""
        mock_response = {
            'message': {
                'content': json.dumps({"x": "Price", "y": "Volume"})
            }
        }
        mock_ollama.chat.return_value = mock_response

        context = "Finance"
        eq = "x + y"

        # We rely on guardrail.validate_mapping being correct.
        # x and y are in eq. Price and Volume are valid terms.

        result = self.translator.map_variables(context, eq)
        self.assertEqual(result, {"x": "Price", "y": "Volume"})

    def test_map_variables_json_repair(self):
        """Test that the translator attempts to repair invalid JSON (single backslashes)."""
        # The LLM returns a string with single backslash for a key which is invalid JSON
        broken_json = r'{"\alpha": "Price"}'
        mock_response = {
            'message': {
                'content': broken_json
            }
        }
        mock_ollama.chat.return_value = mock_response

        context = "Finance"
        eq = r"\alpha"

        # Expect the translator to catch JSONDecodeError, fix backslashes, and succeed
        result = self.translator.map_variables(context, eq)
        self.assertEqual(result.get(r"\alpha"), "Price")

if __name__ == '__main__':
    unittest.main()
