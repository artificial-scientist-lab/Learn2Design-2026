# jagarwal — Round 2 evaluation data

## Summary

- Place: 8 of 60
- Mean feasible loss: 0.161029044
- Sample standard deviation: 0.353245387
- Standard error of the mean: 0.111705999
- Median feasible loss: 0.353835453
- Minimum / maximum run score: -0.409891813 / 0.445569789
- Mean time to best: 81.20 minutes
- Mean evaluations: 138502.4
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
