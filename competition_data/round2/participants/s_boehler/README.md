# s_boehler — Round 2 evaluation data

## Summary

- Place: 10 of 60
- Mean feasible loss: 0.178575800
- Sample standard deviation: 0.225063515
- Standard error of the mean: 0.071171333
- Median feasible loss: 0.217705614
- Minimum / maximum run score: -0.359486349 / 0.409264785
- Mean time to best: 231.90 minutes
- Mean evaluations: 138310.4
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
