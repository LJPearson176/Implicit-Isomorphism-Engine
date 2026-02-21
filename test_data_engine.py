import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# --- Mocking external dependencies before import ---
# We need to mock torch, sentence_transformers, supabase, torch_geometric
# because they are not available in the restricted environment.

mock_torch = MagicMock()
mock_torch.float = 'float'
mock_torch.long = 'long'
sys.modules['torch'] = mock_torch

mock_sentence_transformers = MagicMock()
sys.modules['sentence_transformers'] = mock_sentence_transformers

mock_supabase = MagicMock()
sys.modules['supabase'] = mock_supabase

mock_torch_geometric = MagicMock()
mock_torch_geometric_data = MagicMock()
sys.modules['torch_geometric'] = mock_torch_geometric
sys.modules['torch_geometric.data'] = mock_torch_geometric_data

# Define HeteroData mock specifically to allow attribute access like data['paper'].x
class MockHeteroData:
    def __init__(self):
        self._store = {}
        self._edge_index = {}

    def __getitem__(self, key):
        if isinstance(key, tuple):
            # Accessing edge store
            return self
        if key not in self._store:
            self._store[key] = MagicMock()
        return self._store[key]

    def __setitem__(self, key, value):
         self._store[key] = value

    @property
    def edge_index(self):
        return self._edge_index

    @edge_index.setter
    def edge_index(self, value):
        self._edge_index = value

mock_torch_geometric_data.HeteroData = MockHeteroData

# Now we can import DataEngine
# We need to make sure the import sees our mocks
from data_engine import DataEngine

class TestDataEngine(unittest.TestCase):
    def setUp(self):
        # Setup environment variables
        self.env_patcher = patch.dict(os.environ, {
            "SUPABASE_URL": "http://test-url.com",
            "SUPABASE_KEY": "test-key"
        })
        self.env_patcher.start()

        # Mock create_client
        self.create_client_patcher = patch('data_engine.create_client')
        self.mock_create_client = self.create_client_patcher.start()
        self.mock_supabase_client = MagicMock()
        self.mock_create_client.return_value = self.mock_supabase_client

        # Mock SentenceTransformer
        self.sentence_transformer_patcher = patch('data_engine.SentenceTransformer')
        self.mock_sentence_transformer_cls = self.sentence_transformer_patcher.start()
        self.mock_model = MagicMock()
        self.mock_sentence_transformer_cls.return_value = self.mock_model

        # Initialize DataEngine
        self.engine = DataEngine()

    def tearDown(self):
        self.env_patcher.stop()
        self.create_client_patcher.stop()
        self.sentence_transformer_patcher.stop()

    def test_init_success(self):
        """Test that DataEngine initializes correctly with env vars."""
        self.mock_create_client.assert_called_with("http://test-url.com", "test-key")
        # Should initialize model (either base or fine-tuned, default checks path so likely base if path doesn't exist)
        # We mocked os.path.exists? No, so it depends on real filesystem.
        # Assuming "./models/specter_matchmaker_v1" doesn't exist in test env.
        self.assertTrue(self.mock_sentence_transformer_cls.called)

    def test_init_missing_env(self):
        """Test that ValueError is raised if env vars are missing."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                DataEngine()

    def test_ingest_arxiv_batch_success(self):
        """Test successful ingestion of papers."""
        papers = [
            {'arxiv_id': '1', 'title': 'Paper 1', 'abstract': 'Abstract 1', 'metadata': {}},
            {'arxiv_id': '2', 'title': 'Paper 2', 'abstract': 'Abstract 2', 'metadata': {}}
        ]

        # Mock embedding return
        mock_embeddings = MagicMock()
        mock_embeddings.tolist.side_effect = [[0.1, 0.2], [0.3, 0.4]]
        # We need the encode to return something iterable where each item has tolist?
        # In the code: embeddings = self.model.encode(texts) -> embeddings[i].tolist()
        # So encode should return a list (or array) of objects that have tolist().
        # Or if it returns a tensor, it has tolist() too.
        # Let's just return a list of objects with tolist method.

        class MockEmbedding:
            def __init__(self, data):
                self.data = data
            def tolist(self):
                return self.data

        self.mock_model.encode.return_value = [MockEmbedding([0.1]), MockEmbedding([0.2])]

        # Mock upsert
        mock_table = MagicMock()
        self.mock_supabase_client.table.return_value = mock_table
        mock_upsert = MagicMock()
        mock_table.upsert.return_value = mock_upsert

        # Run method
        self.engine.ingest_arxiv_batch(papers)

        # Assertions
        self.mock_model.encode.assert_called()
        self.mock_supabase_client.table.assert_called_with("papers")
        mock_table.upsert.assert_called()
        args, kwargs = mock_table.upsert.call_args
        self.assertEqual(len(args[0]), 2) # 2 papers
        self.assertEqual(args[0][0]['arxiv_id'], '1')
        mock_upsert.execute.assert_called()

    def test_ingest_arxiv_batch_error(self):
        """Test error handling during ingestion."""
        papers = [{'arxiv_id': '1', 'title': 'T', 'abstract': 'A', 'metadata': {}}]
        self.mock_model.encode.return_value = [MagicMock(tolist=lambda: [0.1])]

        mock_table = MagicMock()
        self.mock_supabase_client.table.return_value = mock_table
        mock_upsert = MagicMock()
        mock_table.upsert.return_value = mock_upsert
        mock_upsert.execute.side_effect = Exception("Upsert failed")

        with self.assertRaises(Exception):
            self.engine.ingest_arxiv_batch(papers)

    def test_build_graph_batch_success(self):
        """Test building graph from papers."""
        paper_ids = ['p1', 'p2']

        # Mock Supabase responses
        # 1. Fetch papers
        mock_papers_resp = MagicMock()
        mock_papers_resp.data = [
            {'id': 'p1', 'embedding': [0.1, 0.1]},
            {'id': 'p2', 'embedding': [0.2, 0.2]}
        ]

        # We need to carefully mock the chain: table().select().in_().execute()
        mock_table = MagicMock()
        self.mock_supabase_client.table.return_value = mock_table
        mock_select = MagicMock()
        mock_table.select.return_value = mock_select
        mock_in = MagicMock()
        mock_select.in_.return_value = mock_in
        mock_in.execute.return_value = mock_papers_resp

        # 2. RPC match_papers (Implicit)
        # Called for each paper.
        # For p1, let's say it matches p2.
        # For p2, no matches.
        mock_rpc_resp_p1 = MagicMock()
        mock_rpc_resp_p1.data = [{'id': 'p2'}]
        mock_rpc_resp_p2 = MagicMock()
        mock_rpc_resp_p2.data = []

        # 3. Citations (Explicit)
        # For p1, cites nothing.
        # For p2, cites p1.
        mock_cite_resp_p1 = MagicMock()
        mock_cite_resp_p1.data = []
        mock_cite_resp_p2 = MagicMock()
        mock_cite_resp_p2.data = [{'target_id': 'p1'}]

        def rpc_side_effect(func, params):
            if func == "match_papers":
                # params['query_embedding'] determines which paper it is
                if params['query_embedding'] == [0.1, 0.1]: # p1
                    return mock_rpc_resp_p1
                else:
                    return mock_rpc_resp_p2
            return MagicMock(data=[])

        self.mock_supabase_client.rpc.side_effect = rpc_side_effect

        # Need to handle table().select().eq().execute() for citations
        # The chain is table("citations").select("target_id").eq("source_id", p_id).execute()

        # We can differentiate calls based on table name
        def table_side_effect(table_name):
            if table_name == "papers":
                return mock_table # configured above for fetching papers
            elif table_name == "citations":
                mock_cite_table = MagicMock()
                mock_cite_select = MagicMock()
                mock_cite_table.select.return_value = mock_cite_select
                mock_cite_eq = MagicMock()
                mock_cite_select.eq.return_value = mock_cite_eq

                def eq_side_effect(col, val):
                    mock_exec = MagicMock()
                    if val == 'p1':
                        mock_exec.execute.return_value = mock_cite_resp_p1
                    elif val == 'p2':
                        mock_exec.execute.return_value = mock_cite_resp_p2
                    return mock_exec

                mock_cite_eq.side_effect = eq_side_effect
                return mock_cite_table
            return MagicMock()

        self.mock_supabase_client.table.side_effect = table_side_effect

        # Run method
        graph = self.engine.build_graph_batch(paper_ids)

        # Assertions
        # Verify torch.tensor was called to create x
        self.assertTrue(mock_torch.tensor.called)

        # Verify graph structure
        # We mocked HeteroData so we can check if edge_index was set
        # Implicit: p1 -> p2 (index 0 -> 1)
        # Explicit: p2 -> p1 (index 1 -> 0)

        # Since we use ThreadPoolExecutor, order of execution isn't guaranteed,
        # but the results should be collected.

        # Check if we got a HeteroData object
        self.assertIsInstance(graph, MockHeteroData)
        self.assertEqual(graph['paper'].num_nodes, 2)

        # We can't easily check the content of edge_index because it's set to a mock_torch.tensor result
        # But we can verify that edge_index was assigned.
        # In build_graph_batch:
        # data['paper', 'cites', 'paper'].edge_index = torch.tensor(explicit_edges, dtype=torch.long)

        # We can check if torch.tensor was called with expected edge lists
        # explicit_edges should be [[1], [0]] (p2 cites p1)
        # implicit_edges should be [[0], [1]] (p1 matches p2)

        # Let's inspect calls to torch.tensor
        # First call is embeddings
        # Then explicit and implicit edges

        # Since calls happen in threads, it's hard to be deterministic about call order or args
        # passed to torch.tensor exactly, but we can check if `supabase` was called correctly.

        # Verify RPC calls
        self.assertEqual(self.mock_supabase_client.rpc.call_count, 2)

        # Verify Citation calls
        # table("citations") called twice
        # We need to check if table was called with "citations"

    def test_build_graph_batch_partial_failure(self):
        """Test build_graph_batch with partial failures in RPC or DB calls."""
        paper_ids = ['p1']

        # Mock Supabase responses for fetching papers
        mock_papers_resp = MagicMock()
        mock_papers_resp.data = [{'id': 'p1', 'embedding': [0.1]}]

        mock_table = MagicMock()
        self.mock_supabase_client.table.return_value = mock_table
        mock_select = MagicMock()
        mock_table.select.return_value = mock_select
        mock_in = MagicMock()
        mock_select.in_.return_value = mock_in
        mock_in.execute.return_value = mock_papers_resp

        # Mock RPC failure
        # It should retry 3 times then fail silently (log only)
        self.mock_supabase_client.rpc.side_effect = Exception("RPC Failed")

        # Mock Citations failure
        # It should retry 3 times then fail silently
        # We need to mock table("citations").select()... raising exception

        def table_side_effect(table_name):
            if table_name == "papers":
                return mock_table
            elif table_name == "citations":
                mock_cite_table = MagicMock()
                mock_cite_table.select.side_effect = Exception("DB Failed")
                return mock_cite_table
            return MagicMock()

        self.mock_supabase_client.table.side_effect = table_side_effect

        # Run method - should not raise exception, but return graph with empty edges
        # We need to mock time.sleep to speed up tests
        with patch('time.sleep', return_value=None):
            graph = self.engine.build_graph_batch(paper_ids)

        self.assertIsInstance(graph, MockHeteroData)
        self.assertEqual(graph['paper'].num_nodes, 1)
        # Should have empty edges
        # In our mock logic, empty edges call torch.empty
        # Verify torch.empty was called twice (once for implicit, once for explicit)
        self.assertEqual(mock_torch.empty.call_count, 2)

if __name__ == "__main__":
    unittest.main()
