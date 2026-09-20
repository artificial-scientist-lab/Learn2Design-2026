# dvolovenko — Round 2 evaluation data

## Summary

- Place: 53 of 60
- Mean feasible loss: 1.980971249
- Sample standard deviation: 1.926188694
- Standard error of the mean: 0.609114348
- Median feasible loss: 0.594075712
- Minimum / maximum run score: 0.418595803 / 4.480464889
- Mean time to best: 130.12 minutes
- Mean evaluations: 55402.2
- Failed runs: 0
- Random-search fallback runs: 0

## Files

- `summary.json`: machine-readable aggregate statistics.
- `runs.csv`: exact final outcome and efficiency statistics for all ten runs.
- `checkpoints.csv`: per-run best feasible loss at selected elapsed times.
- `convergence.csv`: mean and uncertainty across runs at those checkpoints.
- `convergence.png`: visual summary of the convergence data.

Intermediate checkpoint values are marked as exact when the final best was
already known to have been reached. Otherwise they are sampled upper bounds.
The 240-minute values are the exact official run scores.
