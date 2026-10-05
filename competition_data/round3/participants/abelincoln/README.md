# abelincoln — Round 3 evaluation data

## Summary

- Place: 47 of 67
- Mean feasible loss: 0.451941801
- Sample standard deviation: 0.066648482
- Standard error of the mean: 0.021076100
- Median feasible loss: 0.420950059
- Minimum / maximum run score: 0.398810604 / 0.613200898
- Mean time to best: 150.49 minutes
- Mean evaluations: 134197.6
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
