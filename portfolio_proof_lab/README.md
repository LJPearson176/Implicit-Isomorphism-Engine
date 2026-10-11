# Portfolio Proof Lab

Isolated prototype workspace for proof-aware portfolio optimization with deterministic validation and NautilusTrader strategy boundaries.

## Layout

- `configs/`: versioned experiment contract inputs.
- `data/manifests/`: immutable data snapshot metadata and checksums.
- `src/portfolio_proof_lab/research`: signal and covariance estimators.
- `src/portfolio_proof_lab/optimizer`: pure CVXPY optimization service.
- `src/portfolio_proof_lab/verification`: independent allocation validation.
- `src/portfolio_proof_lab/strategy`: scheduled rebalance and order translation boundaries.
- `src/portfolio_proof_lab/reporting`: machine-readable experiment reports.
- `src/portfolio_proof_lab/benchmarks`: controlled benchmark runners.
- `tests/`: unit and integration coverage.
- `experiments/`: generated reports and benchmark artifacts.
- `formal/`: optional Lean-oriented specifications.

## Reproducibility contract

Each run must pin:
- config IDs and experiment ID
- data snapshot checksums
- optimizer solver and tolerances
- software versions
- validation pass/fail checks

## Quickstart

```bash
cd /home/runner/work/Implicit-Isomorphism-Engine/Implicit-Isomorphism-Engine/portfolio_proof_lab
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
pytest -q
```

