import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock modules that are external dependencies or hard to test
sys.modules['torch'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['arxiv'] = MagicMock()
sys.modules['data_engine'] = MagicMock()
sys.modules['models'] = MagicMock()
sys.modules['verify_setup'] = MagicMock()

# Import the module under test
import agent

class TestResearchAgent(unittest.TestCase):
    def setUp(self):
        # Reset mocks between tests
        agent.DataEngine.reset_mock()
        agent.SentenceTransformer.reset_mock()
        agent.WeightedHeteroGraphConv.reset_mock()
        agent.torch.reset_mock()

        # Mock os.path.exists
        self.patcher_exists = patch('os.path.exists')
        self.mock_exists = self.patcher_exists.start()

    def tearDown(self):
        self.patcher_exists.stop()

    def test_init_success(self):
        """Test successful initialization with model loading."""
        self.mock_exists.return_value = True

        # Setup torch.load to return a state dict
        mock_checkpoint = {'model_state_dict': {'key': 'value'}}
        agent.torch.load.return_value = mock_checkpoint

        # Initialize agent
        research_agent = agent.ResearchAgent()

        # Verify components initialized
        agent.DataEngine.assert_called_once()
        agent.SentenceTransformer.assert_called_once_with('allenai/specter')
        agent.WeightedHeteroGraphConv.assert_called_once()

        # Verify model loading
        # agent.torch.load is called twice in current implementation
        agent.torch.load.assert_called_with(agent.MODEL_PATH)
        research_agent.gnn.load_state_dict.assert_called_with(mock_checkpoint['model_state_dict'])
        research_agent.gnn.eval.assert_called_once()

    def test_init_no_model(self):
        """Test initialization when model file does not exist."""
        self.mock_exists.return_value = False

        research_agent = agent.ResearchAgent()

        # Verify components initialized
        agent.DataEngine.assert_called_once()
        agent.SentenceTransformer.assert_called_once()
        agent.WeightedHeteroGraphConv.assert_called_once()

        # Verify no model loading
        agent.torch.load.assert_not_called()
        research_agent.gnn.load_state_dict.assert_not_called()

    def test_query_success(self):
        """Test a successful query flow."""
        self.mock_exists.return_value = False # Avoid model loading logic
        research_agent = agent.ResearchAgent()

        # Mock encoder
        mock_vector = [0.1, 0.2, 0.3]
        research_agent.encoder.encode.return_value = MagicMock(tolist=lambda: mock_vector)

        # Mock Supabase RPC
        mock_candidates = [{'id': 'p1', 'similarity': 0.9}, {'id': 'p2', 'similarity': 0.8}]
        mock_rpc = MagicMock()
        mock_rpc.execute.return_value.data = mock_candidates
        research_agent.engine.supabase.rpc.return_value = mock_rpc

        # Mock Supabase Table Select
        mock_details = [
            {'id': 'p1', 'title': 'Paper 1', 'abstract': 'Abstract 1', 'arxiv_id': '1111'},
            {'id': 'p2', 'title': 'Paper 2', 'abstract': 'Abstract 2', 'arxiv_id': '2222'}
        ]
        mock_select = MagicMock()
        mock_select.in_.return_value.execute.return_value.data = mock_details
        research_agent.engine.supabase.table.return_value.select.return_value = mock_select

        # Call query
        results = research_agent.query("test query")

        # Verify interactions
        research_agent.encoder.encode.assert_called_with("test query")
        research_agent.engine.supabase.rpc.assert_called_with('match_papers', {
            'query_embedding': mock_vector,
            'match_threshold': 0.65,
            'match_count': 15
        })
        research_agent.engine.supabase.table.assert_called_with("papers")

        # Verify results
        self.assertEqual(len(results), 2)
        # Results should be sorted by similarity
        self.assertEqual(results[0]['id'], 'p1')
        self.assertEqual(results[0]['title'], 'Paper 1')
        self.assertEqual(results[1]['id'], 'p2')
        self.assertEqual(results[1]['title'], 'Paper 2')

    def test_query_rpc_fail(self):
        """Test query handles RPC failure gracefully."""
        self.mock_exists.return_value = False
        research_agent = agent.ResearchAgent()

        research_agent.encoder.encode.return_value.tolist.return_value = [0.1]

        # Mock RPC to raise exception
        mock_rpc = MagicMock()
        mock_rpc.execute.side_effect = Exception("RPC Error")
        research_agent.engine.supabase.rpc.return_value = mock_rpc

        results = research_agent.query("test query")

        self.assertIsNone(results)

    def test_query_no_candidates(self):
        """Test query handles no candidates found."""
        self.mock_exists.return_value = False
        research_agent = agent.ResearchAgent()

        research_agent.encoder.encode.return_value.tolist.return_value = [0.1]

        # Mock RPC to return empty list
        mock_rpc = MagicMock()
        mock_rpc.execute.return_value.data = []
        research_agent.engine.supabase.rpc.return_value = mock_rpc

        results = research_agent.query("test query")

        self.assertIsNone(results)
        # Ensure details were not fetched
        research_agent.engine.supabase.table.assert_not_called()

    def test_query_details_fail(self):
        """Test query handles details fetch failure gracefully."""
        self.mock_exists.return_value = False
        research_agent = agent.ResearchAgent()

        research_agent.encoder.encode.return_value.tolist.return_value = [0.1]

        # Mock RPC success
        mock_candidates = [{'id': 'p1', 'similarity': 0.9}]
        mock_rpc = MagicMock()
        mock_rpc.execute.return_value.data = mock_candidates
        research_agent.engine.supabase.rpc.return_value = mock_rpc

        # Mock Table Select failure
        mock_select = MagicMock()
        mock_select.in_.return_value.execute.side_effect = Exception("DB Error")
        research_agent.engine.supabase.table.return_value.select.return_value = mock_select

        results = research_agent.query("test query")

        self.assertIsNone(results)

if __name__ == '__main__':
    unittest.main()
