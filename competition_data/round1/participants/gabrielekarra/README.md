# gabrielekarra — Round 1 evaluation data

## Summary

- Place: 37 of 43
- Mean feasible loss: 1.537802006
- Sample standard deviation: 0.318172390
- Standard error of the mean: 0.100614944
- Median feasible loss: 1.411007337
- Minimum / maximum run score: 1.257793733 / 2.182474077
- Mean time to best: 139.59 minutes
- Mean evaluations: 55168.9
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
