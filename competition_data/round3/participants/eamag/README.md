# eamag — Round 3 evaluation data

## Summary

- Place: 57 of 67
- Mean feasible loss: 0.987366255
- Sample standard deviation: 1.283336495
- Standard error of the mean: 0.405826633
- Median feasible loss: 0.522845799
- Minimum / maximum run score: 0.249528739 / 4.482800322
- Mean time to best: 238.44 minutes
- Mean evaluations: 55305.4
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
