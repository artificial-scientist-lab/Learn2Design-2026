# hamzahanif2210 — Round 2 evaluation data

## Summary

- Place: 30 of 60
- Mean feasible loss: 0.420330425
- Sample standard deviation: 0.174484353
- Standard error of the mean: 0.055176797
- Median feasible loss: 0.426984834
- Minimum / maximum run score: 0.173214245 / 0.752165693
- Mean time to best: 209.35 minutes
- Mean evaluations: 71935.2
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
