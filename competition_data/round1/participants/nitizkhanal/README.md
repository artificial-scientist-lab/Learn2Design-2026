# nitizkhanal — Round 1 evaluation data

## Summary

- Place: 43 of 43
- Mean feasible loss: 5.413418721
- Sample standard deviation: 0.264453615
- Standard error of the mean: 0.083627576
- Median feasible loss: 5.325960029
- Minimum / maximum run score: 5.112641967 / 5.904559765
- Mean time to best: 26.03 minutes
- Mean evaluations: 57945.2
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
derived from one stored candidate every 25 evaluations. The 240-minute values
are the exact official run scores.
