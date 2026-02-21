import unittest
import sys
from unittest.mock import MagicMock, patch

# --- MOCKING DEPENDENCIES BEFORE IMPORT ---
# We must mock torch and numpy because they are not installed in this environment.

# Mock torch
mock_torch = MagicMock()
sys.modules['torch'] = mock_torch

# Mock torch.nn
mock_nn = MagicMock()
sys.modules['torch.nn'] = mock_nn
mock_torch.nn = mock_nn

# Define a MockModule class to allow inheritance
class MockModule:
    def __init__(self):
        pass
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward not implemented in mock")

mock_nn.Module = MockModule

# Mock numpy
mock_numpy = MagicMock()
sys.modules['numpy'] = mock_numpy

# Mock pandas (used in FinanceLoader, though not strictly needed for PhysicsLoss tests,
# but good to mock to prevent import errors if FinanceLoader is initialized)
mock_pandas = MagicMock()
sys.modules['pandas'] = mock_pandas

# Mock verify_setup
sys.modules['verify_setup'] = MagicMock()

# --- IMPORT TARGET MODULE ---
# Now it is safe to import econophysics
import econophysics

class TestPhysicsLoss(unittest.TestCase):
    def setUp(self):
        pass

    def test_default_behavior(self):
        """Test that PhysicsLoss defaults to MSE when generated_loss is missing."""
        # Setup: Mock generated_loss to NOT have custom_physics_loss
        mock_gen_loss = MagicMock()
        del mock_gen_loss.custom_physics_loss

        # Mock MSELoss behavior
        mock_mse_instance = MagicMock()
        # Define what calling the MSE loss instance returns (the loss value)
        # It should return something with an .item() method
        mock_loss_value = MagicMock()
        mock_loss_value.item.return_value = 0.5
        mock_mse_instance.return_value = mock_loss_value

        mock_nn.MSELoss.return_value = mock_mse_instance

        with patch.dict(sys.modules, {'generated_loss': mock_gen_loss}):
            with patch('importlib.reload', return_value=mock_gen_loss):
                # Instantiate
                loss_fn = econophysics.PhysicsLoss()

                # Verify custom_loss is None
                self.assertIsNone(loss_fn.custom_loss)

                # Test forward
                pred = MagicMock()
                target = MagicMock()

                loss = loss_fn(pred, target)

                # Verify MSE was used
                mock_mse_instance.assert_called_with(pred, target)
                self.assertEqual(loss.item(), 0.5)

    def test_custom_loss_behavior(self):
        """Test that PhysicsLoss uses custom_physics_loss when available."""
        # Setup: Mock generated_loss WITH custom_physics_loss
        mock_custom_loss = MagicMock()
        mock_loss_value = MagicMock()
        mock_loss_value.item.return_value = 123.45
        mock_custom_loss.return_value = mock_loss_value

        mock_gen_loss = MagicMock()
        mock_gen_loss.custom_physics_loss = mock_custom_loss

        with patch.dict(sys.modules, {'generated_loss': mock_gen_loss}):
             with patch('importlib.reload', return_value=mock_gen_loss):
                loss_fn = econophysics.PhysicsLoss()

                self.assertIsNotNone(loss_fn.custom_loss)

                pred = MagicMock()
                target = MagicMock()

                loss = loss_fn(pred, target)

                # Verify custom loss was called
                # Note: PhysicsLoss calls custom_loss(predicted_drift, None, None)
                mock_custom_loss.assert_called_with(pred, None, None)
                self.assertEqual(loss.item(), 123.45)

    def test_fallback_behavior(self):
        """Test that PhysicsLoss falls back to MSE if custom_physics_loss fails."""
        # Setup: Mock custom_physics_loss to raise Exception
        mock_custom_loss = MagicMock(side_effect=RuntimeError("AI Error"))

        mock_gen_loss = MagicMock()
        mock_gen_loss.custom_physics_loss = mock_custom_loss

        # Mock MSELoss behavior for fallback
        mock_mse_instance = MagicMock()
        mock_loss_value = MagicMock()
        mock_loss_value.item.return_value = 0.75
        mock_mse_instance.return_value = mock_loss_value

        mock_nn.MSELoss.return_value = mock_mse_instance

        with patch.dict(sys.modules, {'generated_loss': mock_gen_loss}):
             with patch('importlib.reload', return_value=mock_gen_loss):
                loss_fn = econophysics.PhysicsLoss()

                self.assertIsNotNone(loss_fn.custom_loss)

                pred = MagicMock()
                target = MagicMock()

                loss = loss_fn(pred, target)

                # Verify custom loss was tried
                mock_custom_loss.assert_called_with(pred, None, None)

                # Verify fallback to MSE
                mock_mse_instance.assert_called_with(pred, target)
                self.assertEqual(loss.item(), 0.75)

if __name__ == '__main__':
    unittest.main()
