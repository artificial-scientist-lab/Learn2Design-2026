# seantang — Round 3 evaluation data

## Summary

- Place: 24 of 67
- Mean feasible loss: 0.266263428
- Sample standard deviation: 0.113946373
- Standard error of the mean: 0.036033007
- Median feasible loss: 0.284874082
- Minimum / maximum run score: 0.096353998 / 0.408974874
- Mean time to best: 235.40 minutes
- Mean evaluations: 155449.2
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
