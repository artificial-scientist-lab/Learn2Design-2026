# bgkang — Round 1 evaluation data

## Summary

- Place: 18 of 43
- Mean feasible loss: 0.500618700
- Sample standard deviation: 0.103703482
- Standard error of the mean: 0.032793920
- Median feasible loss: 0.476648354
- Minimum / maximum run score: 0.366696325 / 0.640370199
- Mean time to best: 226.46 minutes
- Mean evaluations: 56336.9
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
