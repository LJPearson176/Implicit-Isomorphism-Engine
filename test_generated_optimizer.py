import unittest
import torch
from generated_optimizer import self_optimize_graph

class TestSelfOptimizeGraph(unittest.TestCase):
    def test_warmup_no_pruning(self):
        """Test that for epoch < 5, the original edge index dict is returned unmodified."""
        edge_index = torch.tensor([[0, 1], [1, 0]])
        edge_index_dict = {
            ("paper", "cites", "paper"): edge_index,
            ("author", "wrote_implicit", "paper"): edge_index
        }
        z_dict = {} # Unused
        epoch = 4

        optimized = self_optimize_graph(edge_index_dict, z_dict, epoch)

        # Should return the exact same object
        self.assertIs(optimized, edge_index_dict)

    def test_explicit_edges_preserved(self):
        """Test that explicit edges (no 'implicit' in rel) are cloned and not pruned."""
        edge_index = torch.tensor([[0, 1], [1, 0]])
        edge_index_dict = {
            ("paper", "cites", "paper"): edge_index
        }
        z_dict = {}
        epoch = 100 # Past warmup

        optimized = self_optimize_graph(edge_index_dict, z_dict, epoch)

        # Should be a clone (different object)
        self.assertIsNot(optimized, edge_index_dict)
        self.assertIsNot(optimized[("paper", "cites", "paper")], edge_index)

        # Content should be identical
        self.assertTrue(torch.equal(optimized[("paper", "cites", "paper")], edge_index))

    def test_implicit_edges_pruned(self):
        """Test that implicit edges are pruned based on dropout rate."""
        torch.manual_seed(42)
        num_edges = 1000
        # Create random edges
        edge_index = torch.randint(0, 100, (2, num_edges))
        edge_index_dict = {
            ("author", "wrote_implicit", "paper"): edge_index
        }
        z_dict = {}
        epoch = 100
        # Dropout rate = 0.05 * (100/100) = 0.05
        # Expected kept: ~950

        optimized = self_optimize_graph(edge_index_dict, z_dict, epoch)

        output_edges = optimized[("author", "wrote_implicit", "paper")]
        output_count = output_edges.size(1)

        # Should be strictly less than input (statistically very likely for 1000 edges and 0.05 dropout)
        self.assertLess(output_count, num_edges)

        # With seed 42, we can check specific value if we want, but checking range is safer
        # 0.05 dropout -> keep ~950.
        self.assertGreater(output_count, 900)

    def test_dropout_mechanics(self):
        """Test the dropout rate calculation and clamping."""
        torch.manual_seed(42)
        num_edges = 10000
        edge_index = torch.randint(0, 100, (2, num_edges))
        edge_index_dict = {
            ("author", "wrote_implicit", "paper"): edge_index
        }
        z_dict = {}
        epoch = 600
        # Dropout rate = min(0.05 * 6, 0.3) = 0.3
        # Expected kept: ~7000

        optimized = self_optimize_graph(edge_index_dict, z_dict, epoch)

        output_count = optimized[("author", "wrote_implicit", "paper")].size(1)

        # Check if it's roughly 70%
        ratio = output_count / num_edges
        self.assertAlmostEqual(ratio, 0.7, delta=0.02) # Allow 2% variance

    def test_empty_graph(self):
        """Test with an empty edge index dict."""
        edge_index_dict = {}
        z_dict = {}
        epoch = 100

        optimized = self_optimize_graph(edge_index_dict, z_dict, epoch)

        self.assertEqual(optimized, {})

if __name__ == "__main__":
    unittest.main()
