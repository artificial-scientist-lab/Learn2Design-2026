# rishivg — Round 2 evaluation data

## Summary

- Place: 23 of 60
- Mean feasible loss: 0.343965101
- Sample standard deviation: 0.203183498
- Standard error of the mean: 0.064252264
- Median feasible loss: 0.413137632
- Minimum / maximum run score: -0.219647970 / 0.479058396
- Mean time to best: 168.22 minutes
- Mean evaluations: 55223.7
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
