# actuwien — Round 3 evaluation data

## Summary

- Place: 4 of 67
- Mean feasible loss: -0.211546396
- Sample standard deviation: 0.226342963
- Standard error of the mean: 0.071575930
- Median feasible loss: -0.256732441
- Minimum / maximum run score: -0.493949846 / 0.154597512
- Mean time to best: 237.94 minutes
- Mean evaluations: 164881.5
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
