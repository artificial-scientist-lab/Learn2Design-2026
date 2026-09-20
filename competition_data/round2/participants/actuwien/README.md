# actuwien — Round 2 evaluation data

## Summary

- Place: 1 of 60
- Mean feasible loss: -0.210761363
- Sample standard deviation: 0.203611169
- Standard error of the mean: 0.064387505
- Median feasible loss: -0.259299897
- Minimum / maximum run score: -0.440325457 / 0.148636040
- Mean time to best: 239.64 minutes
- Mean evaluations: 163467.0
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
