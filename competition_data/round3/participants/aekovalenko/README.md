# aekovalenko — Round 3 evaluation data

## Summary

- Place: 5 of 67
- Mean feasible loss: -0.154301892
- Sample standard deviation: 0.295201804
- Standard error of the mean: 0.093351007
- Median feasible loss: -0.233419286
- Minimum / maximum run score: -0.473644255 / 0.347648445
- Mean time to best: 235.43 minutes
- Mean evaluations: 152359.4
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
