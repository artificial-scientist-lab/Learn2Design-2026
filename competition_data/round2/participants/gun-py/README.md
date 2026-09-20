# gun-py — Round 2 evaluation data

## Summary

- Place: 9 of 60
- Mean feasible loss: 0.167875399
- Sample standard deviation: 0.402044135
- Standard error of the mean: 0.127137519
- Median feasible loss: 0.260752187
- Minimum / maximum run score: -0.381480614 / 0.918266018
- Mean time to best: 226.11 minutes
- Mean evaluations: 144952.9
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
