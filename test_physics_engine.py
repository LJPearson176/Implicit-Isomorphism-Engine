import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from physics_engine import PhysicsEngine

class TestPhysicsEngine(unittest.TestCase):
    @patch('physics_engine.Client')
    def setUp(self, mock_client):
        # Prevent network calls during initialization
        self.engine = PhysicsEngine()
        # Verify that client is mocked
        self.assertTrue(isinstance(self.engine.client, MagicMock))

    def test_generate_synthetic_turbulence_basics(self):
        """Test basic functionality with standard input."""
        points = np.array([
            [0, 0, 0],
            [np.pi, 0, 0],
            [np.pi/2, 0, 0]
        ])

        velocities = self.engine._generate_synthetic_turbulence(points)

        # Verify output type and shape
        self.assertIsInstance(velocities, np.ndarray)
        self.assertEqual(velocities.shape, (3, 3))

        # Verify values
        # For p=[0, 0, 0]: u = sin(0) + 0.1*sin(0) = 0
        #                  v = cos(0) + 0.1*cos(0) = 1.1
        #                  w = 0
        self.assertAlmostEqual(velocities[0, 0], 0.0)
        self.assertAlmostEqual(velocities[0, 1], 1.1)
        self.assertAlmostEqual(velocities[0, 2], 0.0)

        # For p=[pi, 0, 0]: u = sin(pi) + 0.1*sin(10*pi) ~ 0
        #                   v = cos(pi) + 0.1*cos(10*pi) = -1 + 0.1*1 = -0.9
        #                   w = 0
        self.assertAlmostEqual(velocities[1, 0], 0.0)
        self.assertAlmostEqual(velocities[1, 1], -0.9)
        self.assertAlmostEqual(velocities[1, 2], 0.0)

    def test_generate_synthetic_turbulence_empty(self):
        """Test with empty input."""
        points = []
        velocities = self.engine._generate_synthetic_turbulence(points)

        self.assertIsInstance(velocities, np.ndarray)
        # Verify shape is (0,) or (0, 3) - np.array([]) creates shape (0,) by default
        self.assertEqual(velocities.shape, (0,))

    def test_generate_synthetic_turbulence_input_types(self):
        """Test with different input types (list of tuples, etc)."""
        # List of lists
        points_list = [[0, 0, 0], [1, 1, 1]]
        velocities = self.engine._generate_synthetic_turbulence(points_list)
        self.assertIsInstance(velocities, np.ndarray)
        self.assertEqual(velocities.shape, (2, 3))

        # List of tuples
        points_tuples = [(0, 0, 0), (1, 1, 1)]
        velocities = self.engine._generate_synthetic_turbulence(points_tuples)
        self.assertIsInstance(velocities, np.ndarray)
        self.assertEqual(velocities.shape, (2, 3))

if __name__ == '__main__':
    unittest.main()
