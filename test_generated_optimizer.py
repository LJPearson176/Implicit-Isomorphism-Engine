import sys
import unittest
import random

# Try to import torch, if not available, define Mock
try:
    import torch
except ImportError:
    # --- MOCKING INFRASTRUCTURE ---
    class MockTensor:
        def __init__(self, data):
            self.data = data # expects list of values, or list of lists for 2D

        def size(self, dim=None):
            if not self.data: return 0
            if isinstance(self.data[0], list): # 2D
                if dim == 0: return len(self.data)
                if dim == 1: return len(self.data[0])
                return (len(self.data), len(self.data[0]))
            else: # 1D
                if dim == 0: return len(self.data)
                return (len(self.data),)

        def clone(self):
            if not self.data: return MockTensor([])
            if isinstance(self.data[0], list):
                return MockTensor([row[:] for row in self.data])
            return MockTensor(self.data[:])

        def __getitem__(self, key):
            if isinstance(key, tuple):
                row_slice, col_slice = key
                if isinstance(col_slice, MockTensor):
                    mask = col_slice.data
                    new_data = []
                    for row in self.data:
                        new_row = [val for val, m in zip(row, mask) if m]
                        new_data.append(new_row)
                    return MockTensor(new_data)
            return self

        def sum(self):
            s = sum(1 for x in self.data if x)
            return MockTensor([s])

        def item(self):
            if isinstance(self.data, list):
                return self.data[0]
            return self.data

        def __gt__(self, other):
            new_data = [x > other for x in self.data]
            return MockTensor(new_data)

        def __repr__(self):
            return f"MockTensor({self.data})"

    class MockTorchModule:
        def tensor(self, data, dtype=None):
            return MockTensor(data)

        def rand(self, size):
            data = [random.random() for _ in range(size)]
            return MockTensor(data)

        def randint(self, low, high, size):
            rows, cols = size
            data = [[random.randint(low, high-1) for _ in range(cols)] for _ in range(rows)]
            return MockTensor(data)

        def manual_seed(self, seed):
            random.seed(seed)

        def equal(self, t1, t2):
            if not isinstance(t1, MockTensor) or not isinstance(t2, MockTensor):
                return False
            return t1.data == t2.data

    sys.modules['torch'] = MockTorchModule()
    import torch

from generated_optimizer import self_optimize_graph

class TestGeneratedOptimizer(unittest.TestCase):
    def setUp(self):
        self.z_dict = {}

    def test_warmup_no_pruning(self):
        """Test that no pruning occurs during warmup epochs (< 5)."""
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        edge_index_dict = {
            ("paper", "cites", "paper"): edge_index,
            ("author", "wrote_implicit", "paper"): edge_index
        }
        epoch = 4

        optimized = self_optimize_graph(edge_index_dict, self.z_dict, epoch)

        self.assertIs(optimized, edge_index_dict)

    def test_explicit_edges_preserved(self):
        """Test that explicit edges are preserved (cloned) and not pruned."""
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        edge_index_dict = {
            ("paper", "cites", "paper"): edge_index
        }
        epoch = 100

        optimized = self_optimize_graph(edge_index_dict, self.z_dict, epoch)

        self.assertTrue(torch.equal(optimized[("paper", "cites", "paper")], edge_index))
        # Ensure it's a clone/copy
        self.assertIsNot(optimized[("paper", "cites", "paper")], edge_index)

    def test_implicit_edges_pruned(self):
        """Test that implicit edges are pruned."""
        torch.manual_seed(42)

        num_edges = 1000
        edge_index = torch.randint(0, 100, (2, num_edges))

        edge_index_dict = {
            ("author", "wrote_implicit", "paper"): edge_index
        }
        epoch = 600 # Max dropout 0.3

        optimized = self_optimize_graph(edge_index_dict, self.z_dict, epoch)

        input_count = edge_index.size(1)
        output_count = optimized[("author", "wrote_implicit", "paper")].size(1)

        self.assertLess(output_count, input_count)
        self.assertTrue(600 < output_count < 800, f"Expected ~700 edges, got {output_count}")

    def test_dropout_mechanics(self):
        """Test different epochs result in different dropout rates."""
        torch.manual_seed(42)
        num_edges = 1000
        edge_index = torch.randint(0, 100, (2, num_edges))
        edge_index_dict = {("author", "wrote_implicit", "paper"): edge_index}

        # Run for epoch 100
        torch.manual_seed(42)
        opt_100 = self_optimize_graph(edge_index_dict, self.z_dict, 100)
        count_100 = opt_100[("author", "wrote_implicit", "paper")].size(1)

        # Run for epoch 600
        torch.manual_seed(42)
        opt_600 = self_optimize_graph(edge_index_dict, self.z_dict, 600)
        count_600 = opt_600[("author", "wrote_implicit", "paper")].size(1)

        self.assertGreater(count_100, count_600,
                           f"Expected fewer edges at higher dropout rate. Got {count_100} vs {count_600}")

    def test_empty_graph(self):
        """Test empty graph handling."""
        edge_index_dict = {}
        epoch = 100
        optimized = self_optimize_graph(edge_index_dict, self.z_dict, epoch)
        self.assertEqual(optimized, {})

if __name__ == "__main__":
    unittest.main()
