# hangglider5 — Round 2 evaluation data

## Summary

- Place: 50 of 60
- Mean feasible loss: 1.316571355
- Sample standard deviation: 1.610354626
- Standard error of the mean: 0.509238846
- Median feasible loss: 0.694678990
- Minimum / maximum run score: 0.196802522 / 4.478312385
- Mean time to best: 170.61 minutes
- Mean evaluations: 55099.5
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
