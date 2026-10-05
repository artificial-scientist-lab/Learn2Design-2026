# lodev — Round 3 evaluation data

## Summary

- Place: 7 of 67
- Mean feasible loss: -0.115564497
- Sample standard deviation: 0.202808086
- Standard error of the mean: 0.064133548
- Median feasible loss: -0.081116098
- Minimum / maximum run score: -0.434897704 / 0.095543791
- Mean time to best: 232.89 minutes
- Mean evaluations: 165742.8
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
