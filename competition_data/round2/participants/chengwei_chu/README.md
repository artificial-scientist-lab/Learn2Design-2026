# chengwei_chu — Round 2 evaluation data

## Summary

- Place: 16 of 60
- Mean feasible loss: 0.261986918
- Sample standard deviation: 0.333460890
- Standard error of the mean: 0.105449592
- Median feasible loss: 0.417290756
- Minimum / maximum run score: -0.423363952 / 0.652369531
- Mean time to best: 235.23 minutes
- Mean evaluations: 152323.4
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
