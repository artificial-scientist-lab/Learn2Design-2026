# danilolapegna — Round 2 evaluation data

## Summary

- Place: 2 of 60
- Mean feasible loss: -0.113979603
- Sample standard deviation: 0.257257041
- Standard error of the mean: 0.081351819
- Median feasible loss: -0.153870366
- Minimum / maximum run score: -0.409908729 / 0.225475168
- Mean time to best: 221.96 minutes
- Mean evaluations: 138794.2
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
