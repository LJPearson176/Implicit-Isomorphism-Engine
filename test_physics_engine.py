import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from physics_engine import PhysicsEngine

class TestPhysicsEngine(unittest.TestCase):

    @patch('physics_engine.Client')
    def test_initialization_success(self, mock_client_cls):
        # Setup mock
        mock_instance = MagicMock()
        mock_client_cls.return_value = mock_instance

        engine = PhysicsEngine()

        self.assertIsNotNone(engine.client)
        self.assertEqual(engine.client, mock_instance)
        mock_client_cls.assert_called_once()

    @patch('physics_engine.Client')
    def test_initialization_failure(self, mock_client_cls):
        # Setup mock to raise exception
        mock_client_cls.side_effect = Exception("Connection failed")

        engine = PhysicsEngine()

        self.assertIsNone(engine.client)

    @patch('physics_engine.Client')
    def test_get_velocity_success(self, mock_client_cls):
        # Setup mock
        mock_instance = MagicMock()
        mock_client_cls.return_value = mock_instance

        # Mock result from GetVelocity
        # result.Vector3 is a list of objects with x, y, z attributes
        mock_vector = MagicMock()
        mock_vector.x = 1.0
        mock_vector.y = 2.0
        mock_vector.z = 3.0

        mock_result = MagicMock()
        mock_result.Vector3 = [mock_vector]

        mock_instance.service.GetVelocity.return_value = mock_result

        engine = PhysicsEngine()

        points = np.array([[0, 0, 0]])
        velocity = engine.get_velocity(points=points)

        self.assertEqual(velocity.shape, (1, 3))
        np.testing.assert_array_equal(velocity, np.array([[1.0, 2.0, 3.0]]))

    @patch('physics_engine.Client')
    def test_get_velocity_failure_fallback(self, mock_client_cls):
        # Setup mock
        mock_instance = MagicMock()
        mock_client_cls.return_value = mock_instance

        # Mock GetVelocity to raise exception
        mock_instance.service.GetVelocity.side_effect = Exception("API Error")

        engine = PhysicsEngine()

        # x = 0
        points = np.array([[0, 0, 0]])
        velocity = engine.get_velocity(points=points)

        # Should fall back to synthetic
        # Synthetic: u = sin(x) + 0.1*sin(10*x), v = cos(x) + 0.1*cos(10*x), w = 0
        # for x=0: u = sin(0) + 0.1*sin(0) = 0
        #          v = cos(0) + 0.1*cos(0) = 1.0 + 0.1 = 1.1
        #          w = 0.0

        self.assertEqual(velocity.shape, (1, 3))
        self.assertAlmostEqual(velocity[0, 0], 0.0)
        self.assertAlmostEqual(velocity[0, 1], 1.1)
        self.assertAlmostEqual(velocity[0, 2], 0.0)

    @patch('physics_engine.Client')
    def test_get_velocity_no_client(self, mock_client_cls):
        # Force client to None
        mock_client_cls.side_effect = Exception("Connection failed")
        engine = PhysicsEngine()
        self.assertIsNone(engine.client)

        points = np.array([[0, 0, 0]])
        velocity = engine.get_velocity(points=points)

        self.assertIsNone(velocity)

    @patch('physics_engine.Client')
    def test_get_velocity_default_points(self, mock_client_cls):
        mock_instance = MagicMock()
        mock_client_cls.return_value = mock_instance

        # Mock default response (20 points expected)
        vectors = []
        for i in range(20):
            v = MagicMock()
            v.x, v.y, v.z = 1.0, 1.0, 1.0
            vectors.append(v)

        mock_result = MagicMock()
        mock_result.Vector3 = vectors
        mock_instance.service.GetVelocity.return_value = mock_result

        engine = PhysicsEngine()
        velocity = engine.get_velocity(points=None)

        self.assertEqual(velocity.shape, (20, 3))
        np.testing.assert_array_equal(velocity, np.ones((20, 3)))

if __name__ == '__main__':
    unittest.main()
