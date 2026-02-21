import sys
import unittest
from unittest.mock import MagicMock, patch, mock_open

# Mock dependencies before importing modules that use them
sys.modules["torch"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["torch_geometric"] = MagicMock()
sys.modules["torch_geometric.data"] = MagicMock()
sys.modules["supabase"] = MagicMock()
sys.modules["arxiv"] = MagicMock()
sys.modules["requests"] = MagicMock()
sys.modules["ollama"] = MagicMock()

# Mock internal modules that might be imported inside functions
# We don't necessarily need to mock them in sys.modules if we patch them where they are used,
# but mocking them here ensures no side effects during import of discovery_engine if it imports them.
# discovery_engine imports agent, fetch_source, translator at top level.
sys.modules["agent"] = MagicMock()
sys.modules["fetch_source"] = MagicMock()
sys.modules["translator"] = MagicMock()
sys.modules["skeptic_agent"] = MagicMock()
sys.modules["compiler"] = MagicMock()

# Now import the module under test
import discovery_engine

class TestDiscoveryPipeline(unittest.TestCase):
    def setUp(self):
        # Reset mocks
        self.mock_agent_cls = discovery_engine.ResearchAgent
        self.mock_agent_instance = self.mock_agent_cls.return_value
        self.mock_agent_instance.reset_mock()

        self.mock_fetch_source = discovery_engine.fetch_source
        self.mock_fetch_source.reset_mock()

        self.mock_extractor_cls = discovery_engine.EquationExtractor
        self.mock_extractor_instance = self.mock_extractor_cls.return_value
        self.mock_extractor_instance.reset_mock()

        self.mock_translator_cls = discovery_engine.NeuroSymbolicTranslator
        self.mock_translator_instance = self.mock_translator_cls.return_value
        self.mock_translator_instance.reset_mock()

        # Determine the skeptic agent class from the mocked module
        # Since we mocked sys.modules['skeptic_agent'], we need to access it from there
        self.mock_skeptic_module = sys.modules["skeptic_agent"]
        self.mock_skeptic_cls = self.mock_skeptic_module.SkepticAgent
        self.mock_skeptic_instance = self.mock_skeptic_cls.return_value
        self.mock_skeptic_instance.reset_mock()

        self.mock_compiler_module = sys.modules["compiler"]
        self.mock_compiler_cls = self.mock_compiler_module.Compiler
        self.mock_compiler_instance = self.mock_compiler_cls.return_value
        self.mock_compiler_instance.reset_mock()

    @patch('discovery_engine.os.walk')
    @patch('builtins.open', new_callable=mock_open, read_data='{"skeptic_enabled": true}')
    def test_happy_path(self, mock_file, mock_walk):
        # Setup Happy Path

        # 1. Agent Retrieval
        self.mock_agent_instance.query.return_value = [{
            'title': 'Test Paper',
            'arxiv_id': '1234.5678',
            'abstract': 'Test Abstract'
        }]

        # 2. Source Ingestion
        self.mock_fetch_source.return_value = '/tmp/source'

        # 3. Equation Extraction
        # Mock os.walk to return a fake file
        mock_walk.return_value = [('/tmp/source', [], ['main.tex'])]

        # Mock extractor to return equations
        self.mock_extractor_instance.extract_from_file.return_value = [
            'E = mc^2',
            '\\frac{d}{dx}f(x) = g(x) + h(x) + k(x) + l(x) + m(x) = 0' # Long equation
        ]

        # 4. Translation
        self.mock_translator_instance.map_variables.return_value = {'E': 'Energy', 'm': 'Mass'}
        self.mock_translator_instance.rewrite_equation.return_value = 'Energy = Mass * c^2'

        # Skeptic
        self.mock_skeptic_instance.critique.return_value = (True, "Looks good")

        # 5. Compilation
        self.mock_translator_instance.generate_code.return_value = 'def energy(m): return m * c**2'
        self.mock_compiler_instance.compile.return_value = True

        # Run Pipeline
        result = discovery_engine.run_discovery_pipeline()

        # Assertions
        self.assertTrue(result['success'])
        self.assertEqual(result['paper_title'], 'Test Paper')
        self.assertEqual(result['arxiv_id'], '1234.5678')
        self.assertIn('Energy = Mass * c^2', result['translated_eq'])
        self.assertEqual(result['generated_code'], 'def energy(m): return m * c**2')

        # Verify calls
        self.mock_agent_instance.query.assert_called_once()
        self.mock_fetch_source.assert_called_with('1234.5678')
        self.mock_extractor_instance.extract_from_file.assert_called()
        self.mock_translator_instance.map_variables.assert_called()
        self.mock_skeptic_instance.critique.assert_called()
        self.mock_compiler_instance.compile.assert_called()

    @patch('discovery_engine.os.walk')
    @patch('builtins.open', new_callable=mock_open, read_data='{"skeptic_enabled": true}')
    def test_agent_failure(self, mock_file, mock_walk):
        # Agent returns empty list
        self.mock_agent_instance.query.return_value = []

        result = discovery_engine.run_discovery_pipeline()

        self.assertFalse(result['success'])
        self.assertIn("No candidates found", result['logs'][1])

    @patch('discovery_engine.os.walk')
    @patch('builtins.open', new_callable=mock_open, read_data='{"skeptic_enabled": true}')
    def test_source_ingestion_failure(self, mock_file, mock_walk):
        # Agent succeeds
        self.mock_agent_instance.query.return_value = [{
            'title': 'Test Paper',
            'arxiv_id': '1234.5678',
            'abstract': 'Test Abstract'
        }]

        # Fetch source fails
        self.mock_fetch_source.return_value = None

        result = discovery_engine.run_discovery_pipeline()

        self.assertFalse(result['success'])
        self.assertIn("Failed to fetch source", result['logs'][-1])

    @patch('discovery_engine.os.walk')
    @patch('builtins.open', new_callable=mock_open, read_data='{"skeptic_enabled": true}')
    def test_equation_extraction_failure(self, mock_file, mock_walk):
        self.mock_agent_instance.query.return_value = [{
            'title': 'Test Paper',
            'arxiv_id': '1234.5678',
            'abstract': 'Test Abstract'
        }]
        self.mock_fetch_source.return_value = '/tmp/source'
        mock_walk.return_value = [('/tmp/source', [], ['main.tex'])]

        # No equations extracted
        self.mock_extractor_instance.extract_from_file.return_value = []

        result = discovery_engine.run_discovery_pipeline()

        self.assertFalse(result['success'])
        # It logs "No equations found at all." if both filtered and fallback are empty
        # Wait, if `all_equations` is empty, `complex_eqs` is empty.
        # Then `if not complex_eqs: log("No complex..."); complex_eqs = all_equations` (still empty).
        # Then `if not complex_eqs: log("No equations found at all."); return`
        self.assertIn("No equations found at all.", result['logs'][-1])

    @patch('discovery_engine.os.walk')
    @patch('builtins.open', new_callable=mock_open, read_data='{"skeptic_enabled": true}')
    def test_translation_failure(self, mock_file, mock_walk):
        self.mock_agent_instance.query.return_value = [{
            'title': 'Test Paper',
            'arxiv_id': '1234.5678',
            'abstract': 'Test Abstract'
        }]
        self.mock_fetch_source.return_value = '/tmp/source'
        mock_walk.return_value = [('/tmp/source', [], ['main.tex'])]
        self.mock_extractor_instance.extract_from_file.return_value = ['long equation with = sign that is long enough']

        # Translation fails (guardrail)
        self.mock_translator_instance.map_variables.return_value = {} # Empty dict acts as falsey in logic?
        # Code: `if not mapping:` -> Circuit breaker.

        result = discovery_engine.run_discovery_pipeline()

        self.assertFalse(result['success'])
        self.assertIn("Neuro-Symbolic Translation Failed", result['logs'][-1])

    @patch('discovery_engine.os.walk')
    @patch('builtins.open', new_callable=mock_open, read_data='{"skeptic_enabled": true}')
    def test_skeptic_rejection(self, mock_file, mock_walk):
        self.mock_agent_instance.query.return_value = [{
            'title': 'Test Paper',
            'arxiv_id': '1234.5678',
            'abstract': 'Test Abstract'
        }]
        self.mock_fetch_source.return_value = '/tmp/source'
        mock_walk.return_value = [('/tmp/source', [], ['main.tex'])]
        self.mock_extractor_instance.extract_from_file.return_value = ['long equation with = sign that is long enough']
        self.mock_translator_instance.map_variables.return_value = {'x': 'y'}
        self.mock_translator_instance.rewrite_equation.return_value = 'y = ...'

        # Skeptic rejects
        self.mock_skeptic_instance.critique.return_value = (False, "Bad mapping")

        result = discovery_engine.run_discovery_pipeline()

        self.assertFalse(result['success'])
        # Code does NOT log to result_payload["logs"] for skeptic, it prints directly.
        # But it returns result_payload.
        # I should capture stdout to verify the print, or just verify success is False.
        # The code:
        # if not approved:
        #     print(f"❌ SKEPTIC REJECTED MAPPING: {critique}")
        #     print("Terminating pipeline to prevent hallucination.")
        #     return result_payload
        # success is initialized to False.

        self.assertFalse(result['success'])

    @patch('discovery_engine.os.walk')
    @patch('builtins.open', new_callable=mock_open, read_data='{"skeptic_enabled": true}')
    def test_compilation_failure(self, mock_file, mock_walk):
        self.mock_agent_instance.query.return_value = [{
            'title': 'Test Paper',
            'arxiv_id': '1234.5678',
            'abstract': 'Test Abstract'
        }]
        self.mock_fetch_source.return_value = '/tmp/source'
        mock_walk.return_value = [('/tmp/source', [], ['main.tex'])]
        self.mock_extractor_instance.extract_from_file.return_value = ['long equation with = sign that is long enough']
        self.mock_translator_instance.map_variables.return_value = {'x': 'y'}
        self.mock_skeptic_instance.critique.return_value = (True, "OK")
        self.mock_translator_instance.generate_code.return_value = 'code'

        # Compilation fails
        self.mock_compiler_instance.compile.return_value = False

        result = discovery_engine.run_discovery_pipeline()

        self.assertFalse(result['success'])
        self.assertTrue(any("Compilation Failed" in log for log in result['logs']))

if __name__ == '__main__':
    unittest.main()
