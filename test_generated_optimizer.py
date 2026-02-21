import sys
import unittest
from unittest.mock import patch
import importlib

# Import our mock implementation
import mock_torch_impl

class TestGeneratedOptimizer(unittest.TestCase):

    def setUp(self):
        # Patch sys.modules so 'import torch' resolves to our mock
        self.patcher = patch.dict(sys.modules, {'torch': mock_torch_impl})
        self.patcher.start()

        # Ensure generated_optimizer is imported with the mock torch
        if 'generated_optimizer' in sys.modules:
            del sys.modules['generated_optimizer']

        import generated_optimizer
        self.optimizer_module = generated_optimizer

    def tearDown(self):
        self.patcher.stop()
        # Clean up generated_optimizer from sys.modules
        if 'generated_optimizer' in sys.modules:
            del sys.modules['generated_optimizer']

    def test_warmup_no_pruning(self):
        """Test that no pruning occurs during warmup (epoch < 5)."""
        edge_index = mock_torch_impl.tensor([[0, 1, 2], [1, 2, 0]])
        edge_index_dict = {
            ("paper", "cites", "paper"): edge_index,
            ("author", "wrote_implicit", "paper"): edge_index
        }
        z_dict = {}
        epoch = 4

        optimized = self.optimizer_module.self_optimize_graph(edge_index_dict, z_dict, epoch)

        # Check identity
        self.assertIs(optimized, edge_index_dict)

    def test_explicit_edges_preserved(self):
        """Test that explicit edges are preserved (cloned) after warmup."""
        edge_index = mock_torch_impl.tensor([[0, 1, 2], [1, 2, 0]])
        edge_index_dict = {
            ("paper", "cites", "paper"): edge_index
        }
        z_dict = {}
        epoch = 100

        optimized = self.optimizer_module.self_optimize_graph(edge_index_dict, z_dict, epoch)

        # Check value equality
        self.assertTrue(mock_torch_impl.equal(optimized[("paper", "cites", "paper")], edge_index))
        # Check it is a clone (different object)
        self.assertIsNot(optimized[("paper", "cites", "paper")], edge_index)

    def test_implicit_edges_pruned(self):
        """Test that implicit edges are pruned based on dropout rate."""
        mock_torch_impl.manual_seed(42)

        num_edges = 1000
        edge_index = mock_torch_impl.randint(0, 100, (2, num_edges))

        edge_index_dict = {
            ("author", "wrote_implicit", "paper"): edge_index
        }
        z_dict = {}
        epoch = 600 # Max dropout 0.3

        optimized = self.optimizer_module.self_optimize_graph(edge_index_dict, z_dict, epoch)

        input_count = edge_index.size(1)
        output_count = optimized[("author", "wrote_implicit", "paper")].size(1)

        print(f"Input edges: {input_count}, Output edges: {output_count}")

        self.assertLess(output_count, input_count)
        self.assertTrue(650 < output_count < 750)

    def test_dropout_scaling(self):
        """Test dropout scaling logic."""
        # epoch=100 -> dropout 0.05
        # epoch=200 -> dropout 0.10

        num_edges = 10000
        edge_index = mock_torch_impl.randint(0, 100, (2, num_edges))
        edge_index_dict = {("author", "wrote_implicit", "paper"): edge_index}
        z_dict = {}

        # Epoch 100
        mock_torch_impl.manual_seed(42)
        opt_100 = self.optimizer_module.self_optimize_graph(edge_index_dict, z_dict, 100)
        count_100 = opt_100[("author", "wrote_implicit", "paper")].size(1)

        # Epoch 200
        mock_torch_impl.manual_seed(42)
        opt_200 = self.optimizer_module.self_optimize_graph(edge_index_dict, z_dict, 200)
        count_200 = opt_200[("author", "wrote_implicit", "paper")].size(1)

        # Expect more pruning at epoch 200
        self.assertLess(count_200, count_100)

if __name__ == "__main__":
    unittest.main()
