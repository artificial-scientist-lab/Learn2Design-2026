# bgkang — Round 2 evaluation data

## Summary

- Place: 14 of 60
- Mean feasible loss: 0.244625610
- Sample standard deviation: 0.104982059
- Standard error of the mean: 0.033198242
- Median feasible loss: 0.196024038
- Minimum / maximum run score: 0.150477297 / 0.411968812
- Mean time to best: 211.46 minutes
- Mean evaluations: 86128.7
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
