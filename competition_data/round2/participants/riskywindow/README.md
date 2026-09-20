# riskywindow — Round 2 evaluation data

## Summary

- Place: 56 of 60
- Mean feasible loss: 2.763186797
- Sample standard deviation: 0.329257462
- Standard error of the mean: 0.104120352
- Median feasible loss: 2.777898840
- Minimum / maximum run score: 2.246392220 / 3.393492101
- Mean time to best: 157.76 minutes
- Mean evaluations: 56369.3
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
