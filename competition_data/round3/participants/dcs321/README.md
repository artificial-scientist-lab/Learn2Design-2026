# dcs321 — Round 3 evaluation data

## Summary

- Place: 29 of 67
- Mean feasible loss: 0.307334841
- Sample standard deviation: 0.153457603
- Standard error of the mean: 0.048527555
- Median feasible loss: 0.354095292
- Minimum / maximum run score: -0.048155563 / 0.466397567
- Mean time to best: 236.83 minutes
- Mean evaluations: 153526.4
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
