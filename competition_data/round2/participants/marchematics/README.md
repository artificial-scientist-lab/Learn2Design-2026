# marchematics — Round 2 evaluation data

## Summary

- Place: 41 of 60
- Mean feasible loss: 0.656706752
- Sample standard deviation: 0.297923845
- Standard error of the mean: 0.094211792
- Median feasible loss: 0.593387715
- Minimum / maximum run score: 0.192855798 / 1.132564900
- Mean time to best: 238.98 minutes
- Mean evaluations: 136325.5
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
