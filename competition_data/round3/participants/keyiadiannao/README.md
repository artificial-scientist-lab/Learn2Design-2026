# keyiadiannao — Round 3 evaluation data

## Summary

- Place: 38 of 67
- Mean feasible loss: 0.367578951
- Sample standard deviation: 0.084104255
- Standard error of the mean: 0.026596101
- Median feasible loss: 0.410940879
- Minimum / maximum run score: 0.227984859 / 0.446285056
- Mean time to best: 214.34 minutes
- Mean evaluations: 133690.8
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
