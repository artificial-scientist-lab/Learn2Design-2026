# hendrik6687 — Round 3 evaluation data

## Summary

- Place: 27 of 67
- Mean feasible loss: 0.300784814
- Sample standard deviation: 0.143966487
- Standard error of the mean: 0.045526201
- Median feasible loss: 0.336089068
- Minimum / maximum run score: -0.053014379 / 0.413514405
- Mean time to best: 238.37 minutes
- Mean evaluations: 130512.4
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
