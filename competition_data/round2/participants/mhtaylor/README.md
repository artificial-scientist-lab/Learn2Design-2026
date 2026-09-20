# mhtaylor — Round 2 evaluation data

## Summary

- Place: 3 of 60
- Mean feasible loss: -0.017196807
- Sample standard deviation: 0.133758743
- Standard error of the mean: 0.042298228
- Median feasible loss: -0.051909927
- Minimum / maximum run score: -0.196707354 / 0.194433538
- Mean time to best: 213.18 minutes
- Mean evaluations: 150246.4
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
