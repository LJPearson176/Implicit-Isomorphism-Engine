
import sys
import unittest
from unittest.mock import MagicMock, mock_open, patch

# --- Mock Heavy Dependencies ---
# Mock sys.modules for torch, requests, verify_setup before importing translator
# This prevents the actual modules from being loaded and avoids ImportError/SystemExit
sys.modules["torch"] = MagicMock()
sys.modules["requests"] = MagicMock()

# Mock verify_setup
# verify_setup.py executes code at module level (load_env()) and imports DataEngine.
# We mock it completely to prevent side effects.
mock_verify_setup = MagicMock()
# Ensure load_env() exists on the mock since translator imports it
mock_verify_setup.load_env = MagicMock()
sys.modules["verify_setup"] = mock_verify_setup

# Now import the class under test
from translator import EquationExtractor

class TestEquationExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = EquationExtractor()

    def test_extract_equation_env(self):
        r"""Test extraction of \begin{equation} ... \end{equation} block."""
        content = r"""
        Here is some text.
        \begin{equation}
            E = mc^2
        \end{equation}
        More text.
        """
        with patch("builtins.open", mock_open(read_data=content)):
            equations = self.extractor.extract_from_file("dummy.tex")
            self.assertEqual(len(equations), 1)
            # The regex captures the content inside the environment
            self.assertIn("E = mc^2", equations[0])

    def test_extract_double_dollar(self):
        """Test extraction of $$ ... $$ block."""
        content = r"""
        Text before.
        $$
        a^2 + b^2 = c^2
        $$
        Text after.
        """
        with patch("builtins.open", mock_open(read_data=content)):
            equations = self.extractor.extract_from_file("dummy.tex")
            self.assertEqual(len(equations), 1)
            self.assertIn("a^2 + b^2 = c^2", equations[0])

    def test_extract_bracket_notation(self):
        r"""Test extraction of \[ ... \] block."""
        content = r"""
        \[ \frac{1}{2} \]
        """
        with patch("builtins.open", mock_open(read_data=content)):
            equations = self.extractor.extract_from_file("dummy.tex")
            self.assertEqual(len(equations), 1)
            self.assertIn(r"\frac{1}{2}", equations[0])

    def test_extract_aligned(self):
        r"""Test extraction of \begin{aligned} ... \end{aligned} block."""
        content = r"""
        \begin{aligned}
            x &= y \\
            y &= z
        \end{aligned}
        """
        with patch("builtins.open", mock_open(read_data=content)):
            equations = self.extractor.extract_from_file("dummy.tex")
            self.assertEqual(len(equations), 1)
            self.assertIn(r"x &= y \\", equations[0])

    def test_multiple_equations(self):
        """Test extraction of multiple equations in a single file."""
        content = r"""
        First equation:
        \begin{equation}
            eq1
        \end{equation}

        Second equation:
        $$ eq2 $$

        Third equation:
        \[ eq3 \]
        """
        with patch("builtins.open", mock_open(read_data=content)):
            equations = self.extractor.extract_from_file("dummy.tex")
            self.assertEqual(len(equations), 3)
            # Order depends on regex match order if patterns are checked sequentially.
            # EquationExtractor iterates over patterns one by one.
            # Patterns order: equation, $$, \[, aligned

            # Since patterns are checked in order, 'eq1' (equation) comes first.
            # Then 'eq2' ($$), then 'eq3' (\[).
            # If they appeared in different order in file, the list order would follow pattern order, not file order.
            # Let's verify presence primarily.
            self.assertTrue(any("eq1" in eq for eq in equations))
            self.assertTrue(any("eq2" in eq for eq in equations))
            self.assertTrue(any("eq3" in eq for eq in equations))

    def test_nested_blocks(self):
        """Test nested blocks behavior (e.g. aligned inside equation)."""
        content = r"""
        \begin{equation}
            \begin{aligned}
                x &= y
            \end{aligned}
        \end{equation}
        """
        with patch("builtins.open", mock_open(read_data=content)):
            equations = self.extractor.extract_from_file("dummy.tex")
            # Should match outer 'equation' block AND inner 'aligned' block
            # because patterns are independent.
            self.assertEqual(len(equations), 2)

            # Verify the aligned block content is present (extracted by aligned pattern)
            aligned_match = [eq for eq in equations if "x &= y" in eq and "begin{aligned}" not in eq]
            self.assertTrue(len(aligned_match) > 0, "Inner aligned content not extracted cleanly")

            # Verify the equation block content (which includes the aligned block string)
            equation_match = [eq for eq in equations if "begin{aligned}" in eq]
            self.assertTrue(len(equation_match) > 0, "Outer equation block not extracted")

    def test_no_equations(self):
        """Test file with no equations returns empty list."""
        content = "Just some plain text with no math."
        with patch("builtins.open", mock_open(read_data=content)):
            equations = self.extractor.extract_from_file("dummy.tex")
            self.assertEqual(equations, [])

    def test_empty_file(self):
        """Test completely empty file."""
        content = ""
        with patch("builtins.open", mock_open(read_data=content)):
            equations = self.extractor.extract_from_file("dummy.tex")
            self.assertEqual(equations, [])

    def test_file_error(self):
        """Test handling of file read errors."""
        with patch("builtins.open", side_effect=IOError("File not found")):
            # Should catch exception and print error (we mock print to suppress output?)
            # The method returns [] on exception.
            equations = self.extractor.extract_from_file("nonexistent.tex")
            self.assertEqual(equations, [])

if __name__ == "__main__":
    unittest.main()
