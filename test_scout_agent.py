import sys
from unittest.mock import MagicMock

# Mock external dependencies that are not installed
sys.modules['arxiv'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['requests'] = MagicMock()
sys.modules['bs4'] = MagicMock()
sys.modules['supabase'] = MagicMock()
sys.modules['torch_geometric'] = MagicMock()
sys.modules['torch_geometric.data'] = MagicMock()

import unittest
from unittest.mock import patch
import scout_agent

class TestScoutAgent(unittest.TestCase):

    @patch('scout_agent.arxiv.Search')
    @patch('scout_agent.arxiv.Client')
    @patch('scout_agent.compiler.Compiler')
    def test_run_scout_mission_success(self, mock_compiler_cls, mock_client_cls, mock_search_cls):
        # Setup mocks
        mock_client = mock_client_cls.return_value
        mock_compiler = mock_compiler_cls.return_value

        # Mock paper
        mock_paper = MagicMock()
        mock_paper.title = "Test Paper Title"
        mock_paper.summary = "Test Paper Abstract\nWith Newlines"

        # Mock search results
        mock_client.results.return_value = [mock_paper]

        # Mock compiler
        mock_compiler.compile_isomorphism.return_value = "generated_scout_solution.py"

        # Run function
        scout_agent.run_scout_mission()

        # Assertions
        mock_search_cls.assert_called_once()
        mock_client.results.assert_called_once()
        mock_compiler_cls.assert_called_once()
        mock_compiler.compile_isomorphism.assert_called_once()

        # Verify bridge_edge content passed to compiler
        call_args = mock_compiler.compile_isomorphism.call_args
        bridge_edge = call_args[0][0]
        self.assertEqual(bridge_edge['source_paper'], "Test Paper Title")
        self.assertEqual(bridge_edge['target_domain'], "Biology (Protein Folding)")

    @patch('scout_agent.arxiv.Search')
    @patch('scout_agent.arxiv.Client')
    @patch('scout_agent.compiler.Compiler')
    def test_run_scout_mission_no_papers(self, mock_compiler_cls, mock_client_cls, mock_search_cls):
        # Setup mocks
        mock_client = mock_client_cls.return_value
        mock_client.results.return_value = [] # No papers

        # Run function
        scout_agent.run_scout_mission()

        # Assertions
        mock_search_cls.assert_called_once()
        mock_client.results.assert_called_once()
        mock_compiler_cls.assert_not_called() # Compiler should not be instantiated or used

    @patch('scout_agent.arxiv.Search')
    @patch('scout_agent.arxiv.Client')
    @patch('scout_agent.compiler.Compiler')
    def test_run_scout_mission_compile_failure(self, mock_compiler_cls, mock_client_cls, mock_search_cls):
        # Setup mocks
        mock_client = mock_client_cls.return_value
        mock_compiler = mock_compiler_cls.return_value

        # Mock paper
        mock_paper = MagicMock()
        mock_paper.title = "Test Paper Title"
        mock_paper.summary = "Test Paper Abstract"

        # Mock search results
        mock_client.results.return_value = [mock_paper]

        # Mock compiler failure
        mock_compiler.compile_isomorphism.return_value = None

        # Run function
        scout_agent.run_scout_mission()

        # Assertions
        mock_search_cls.assert_called_once()
        mock_client.results.assert_called_once()
        mock_compiler_cls.assert_called_once()
        mock_compiler.compile_isomorphism.assert_called_once()

if __name__ == '__main__':
    unittest.main()
