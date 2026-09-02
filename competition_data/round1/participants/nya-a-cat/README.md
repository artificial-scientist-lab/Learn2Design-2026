# nya-a-cat — Round 1 evaluation data

## Summary

- Place: 24 of 43
- Mean feasible loss: 0.617681999
- Sample standard deviation: 0.438743033
- Standard error of the mean: 0.138742729
- Median feasible loss: 0.423470277
- Minimum / maximum run score: 0.144716149 / 1.734657193
- Mean time to best: 234.70 minutes
- Mean evaluations: 142765.9
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
