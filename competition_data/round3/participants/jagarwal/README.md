# jagarwal — Round 3 evaluation data

## Summary

- Place: 8 of 67
- Mean feasible loss: -0.098427524
- Sample standard deviation: 0.332620773
- Standard error of the mean: 0.105183924
- Median feasible loss: -0.292344477
- Minimum / maximum run score: -0.390286808 / 0.433744474
- Mean time to best: 158.96 minutes
- Mean evaluations: 139257.6
- Failed runs: 0
- Random-search fallback runs: 0

## Files

- `summary.json`: machine-readable aggregate statistics.
- `runs.csv`: exact final outcome and efficiency statistics for all ten runs.
- `checkpoints.csv`: per-run best feasible loss at selected elapsed times.
- `convergence.csv`: mean and uncertainty across runs at those checkpoints.
- `convergence.png`: visual summary of the convergence data.

Intermediate checkpoints are conservative upper bounds unless marked exact.
The 240-minute values are official run scores; missing values and fallbacks
are marked.
