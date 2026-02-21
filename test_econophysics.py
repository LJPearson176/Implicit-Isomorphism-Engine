import unittest
from unittest.mock import MagicMock, patch
import sys

# 1. Mock external dependencies
mock_torch = MagicMock()
mock_torch.float32 = 'float32'
sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = MagicMock()

mock_numpy = MagicMock()
sys.modules['numpy'] = mock_numpy

mock_pandas = MagicMock()
sys.modules['pandas'] = mock_pandas

# Mock verify_setup
sys.modules['verify_setup'] = MagicMock()

# 2. Import module under test
import econophysics
from econophysics import FinanceLoader

class TestFinanceLoader(unittest.TestCase):
    def setUp(self):
        self.loader = FinanceLoader(n_points=100, dt=0.01)

    @patch('econophysics.os.path.exists')
    def test_load_real_data_success(self, mock_exists):
        # Setup mock for os.path.exists
        mock_exists.return_value = True

        # Setup mock for pandas (imported inside the function)
        # Since we mocked sys.modules['pandas'], the 'import pandas as pd' inside the function returns our mock_pandas
        mock_read_csv = mock_pandas.read_csv

        # Create a mock DataFrame structure
        mock_df = MagicMock()
        mock_read_csv.return_value = mock_df

        # Configure df["T"]
        mock_series_T = MagicMock()

        # Configure df["pressure"]
        mock_series_P = MagicMock()
        mock_series_P.values = [100.0, 101.0]

        # Configure df[...] access
        def df_getitem(key):
            if key == "T":
                return mock_series_T
            elif key == "pressure":
                return mock_series_P
            return MagicMock()
        mock_df.__getitem__.side_effect = df_getitem

        # Configure (df["T"] - df["T"].min()).dt.total_seconds().values
        mock_min_val = MagicMock()
        mock_series_T.min.return_value = mock_min_val

        mock_sub_res = MagicMock()
        mock_series_T.__sub__.return_value = mock_sub_res # For df["T"] - min

        mock_dt_accessor = MagicMock()
        mock_sub_res.dt = mock_dt_accessor

        mock_total_seconds = MagicMock()
        mock_dt_accessor.total_seconds.return_value = mock_total_seconds

        # Returns [0, dt] so we can test dt calculation
        # Let's say dt = 0.5
        mock_total_seconds.values = [0.0, 0.5]

        # Configure torch.tensor().view()
        # The code does: t = torch.tensor(t_raw, ...).view(-1, 1)
        # We need to ensure that t has length > 1 for the dt calculation check

        # We can mock the return value of view()
        mock_tensor_t = MagicMock()
        mock_tensor_t.__len__.return_value = 2 # len(t) > 1

        mock_tensor_P = MagicMock()

        # We need to distinguish between the two torch.tensor calls if possible, or make the mock generic enough
        # 1. t_raw -> torch.tensor -> view
        # 2. P_raw -> torch.tensor -> view

        # side_effect for torch.tensor
        def tensor_side_effect(data, dtype=None):
            m = MagicMock()
            if data is mock_total_seconds.values:
                # This is 't'
                m.view.return_value = mock_tensor_t
            elif data is mock_series_P.values:
                # This is 'P'
                m.view.return_value = mock_tensor_P
            else:
                m.view.return_value = MagicMock()
            return m

        mock_torch.tensor.side_effect = tensor_side_effect

        # dt calculation: self.dt = (t[1] - t[0]).item()
        # t is mock_tensor_t

        mock_t0 = MagicMock()
        mock_t1 = MagicMock()

        def t_getitem(idx):
            if idx == 0: return mock_t0
            if idx == 1: return mock_t1
            return MagicMock()

        mock_tensor_t.__getitem__.side_effect = t_getitem

        mock_diff = MagicMock()
        mock_t1.__sub__.return_value = mock_diff
        mock_diff.item.return_value = 0.5

        # --- Execution ---
        filepath = "test_lob_data.csv"
        t, P = self.loader.load_real_data(filepath=filepath)

        # --- Verification ---
        mock_exists.assert_called_with(filepath)
        mock_read_csv.assert_called_with(filepath, parse_dates=["T"])

        self.assertEqual(t, mock_tensor_t)
        self.assertEqual(P, mock_tensor_P)
        self.assertEqual(self.loader.dt, 0.5)

    @patch('econophysics.os.path.exists')
    def test_load_real_data_file_not_found(self, mock_exists):
        mock_exists.return_value = False
        t, P = self.loader.load_real_data(filepath="missing.csv")
        mock_exists.assert_called_with("missing.csv")
        self.assertIsNone(t)
        self.assertIsNone(P)

    def test_generate_synthetic_data(self):
        # Mock numpy functions used
        # t = np.linspace(0, self.n_points*self.dt, self.n_points)
        mock_linspace_res = MagicMock()
        mock_numpy.linspace.return_value = mock_linspace_res

        # W = np.random.standard_normal(size=self.n_points)
        mock_std_norm_res = MagicMock()
        mock_numpy.random.standard_normal.return_value = mock_std_norm_res

        # W = np.cumsum(W)*np.sqrt(self.dt)
        mock_cumsum_res = MagicMock()
        mock_numpy.cumsum.return_value = mock_cumsum_res

        mock_sqrt_res = MagicMock()
        mock_numpy.sqrt.return_value = mock_sqrt_res

        mock_w_final = MagicMock()
        # W = cumsum * sqrt
        mock_cumsum_res.__mul__.return_value = mock_w_final

        # X = (mu - 0.5*sigma**2)*t + sigma*W
        # S = S0 * np.exp(X)
        mock_exp_res = MagicMock()
        mock_numpy.exp.return_value = mock_exp_res

        # Check return values (torch tensors)
        mock_tensor_t = MagicMock()
        mock_tensor_S = MagicMock()

        # torch.tensor(t, ...).view(...)
        # torch.tensor(S, ...).view(...)

        def tensor_side_effect(data, dtype=None):
            m = MagicMock()
            if data is mock_linspace_res:
                m.view.return_value = mock_tensor_t
            elif data is mock_exp_res: # S is constructed from np.exp(X) * S0, but effectively depends on exp
                # Wait, S = S0 * np.exp(X). So data passed to torch.tensor is S.
                pass
            return m

        # Since checking exact object identity for S might be tricky due to arithmetic operations creating new mocks,
        # we can just return generic mocks and verify torch.tensor was called.

        mock_torch.tensor.return_value.view.return_value = MagicMock()

        t, P = self.loader.generate_synthetic_data()

        # Verify calls
        mock_numpy.linspace.assert_called()
        mock_numpy.random.standard_normal.assert_called_with(size=self.loader.n_points)
        self.assertEqual(mock_torch.tensor.call_count, 2)

        # Verify we got something back
        self.assertTrue(isinstance(t, MagicMock))
        self.assertTrue(isinstance(P, MagicMock))

if __name__ == '__main__':
    unittest.main()
