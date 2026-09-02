# matin — Round 1 evaluation data

## Summary

- Place: 12 of 43
- Mean feasible loss: 0.429490613
- Sample standard deviation: 0.126121602
- Standard error of the mean: 0.039883152
- Median feasible loss: 0.443803816
- Minimum / maximum run score: 0.205032664 / 0.624923232
- Mean time to best: 195.87 minutes
- Mean evaluations: 132782.0
- Failed runs: 0
- Random-search fallback runs: 0

## Files

- `summary.json`: machine-readable aggregate statistics.
- `runs.csv`: exact final outcome and efficiency statistics for all ten runs.
- `checkpoints.csv`: per-run best feasible loss at selected elapsed times.
- `convergence.csv`: mean and uncertainty across runs at those checkpoints.
- `convergence.png`: visual summary of the convergence data.

Intermediate checkpoint values are marked as exact when the final best was
already known to have been reached. Otherwise they are sampled upper bounds
derived from one stored candidate every 25 evaluations. The 240-minute values
are the exact official run scores.
