# akira-higuchi — Round 3 evaluation data

## Summary

- Place: 65 of 67
- Mean feasible loss: 2.158224709
- Sample standard deviation: 1.761979438
- Standard error of the mean: 0.557186821
- Median feasible loss: 1.146643816
- Minimum / maximum run score: 0.411403635 / 4.503845195
- Mean time to best: 136.40 minutes
- Mean evaluations: 56530.4
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
