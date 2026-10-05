# gun-py — Round 3 evaluation data

## Summary

- Place: 18 of 67
- Mean feasible loss: 0.203149360
- Sample standard deviation: 0.173695869
- Standard error of the mean: 0.054927457
- Median feasible loss: 0.211302691
- Minimum / maximum run score: -0.149515773 / 0.404245506
- Mean time to best: 208.96 minutes
- Mean evaluations: 90252.8
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
