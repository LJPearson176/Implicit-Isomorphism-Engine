import unittest

import pandas as pd

from portfolio_proof_lab.research.estimators import assert_no_lookahead


class TestTimeIntegrity(unittest.TestCase):
    def test_rejects_lookahead(self):
        decision_time = pd.Timestamp("2026-01-01T10:00:00Z")
        feature_times = pd.Series(
            [
                pd.Timestamp("2026-01-01T09:59:00Z"),
                pd.Timestamp("2026-01-01T10:01:00Z"),
            ]
        )
        with self.assertRaises(ValueError):
            assert_no_lookahead(feature_times, decision_time)

    def test_accepts_past_only(self):
        decision_time = pd.Timestamp("2026-01-01T10:00:00Z")
        feature_times = pd.Series(
            [
                pd.Timestamp("2026-01-01T09:58:00Z"),
                pd.Timestamp("2026-01-01T09:59:00Z"),
            ]
        )
        assert_no_lookahead(feature_times, decision_time)


if __name__ == "__main__":
    unittest.main()
