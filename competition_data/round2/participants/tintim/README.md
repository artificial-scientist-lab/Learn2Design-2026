# tintim — Round 2 evaluation data

## Summary

- Place: 39 of 60
- Mean feasible loss: 0.634455105
- Sample standard deviation: 0.161680802
- Standard error of the mean: 0.051127959
- Median feasible loss: 0.663213036
- Minimum / maximum run score: 0.407168222 / 0.844504815
- Mean time to best: 175.66 minutes
- Mean evaluations: 146203.1
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
derived from one stored candidate every 10 evaluations. The 240-minute values
are the exact official run scores.
