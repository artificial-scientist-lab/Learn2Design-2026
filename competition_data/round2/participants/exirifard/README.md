# exirifard — Round 2 evaluation data

## Summary

- Place: 18 of 60
- Mean feasible loss: 0.297833766
- Sample standard deviation: 0.166758783
- Standard error of the mean: 0.052733758
- Median feasible loss: 0.353153224
- Minimum / maximum run score: -0.125476243 / 0.421857142
- Mean time to best: 236.92 minutes
- Mean evaluations: 132021.2
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
