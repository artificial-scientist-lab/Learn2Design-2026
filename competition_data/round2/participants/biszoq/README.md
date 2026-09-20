# biszoq — Round 2 evaluation data

## Summary

- Place: 52 of 60
- Mean feasible loss: 1.486363372
- Sample standard deviation: 0.727437063
- Standard error of the mean: 0.230035797
- Median feasible loss: 1.625734561
- Minimum / maximum run score: 0.409248612 / 2.493260309
- Mean time to best: 220.53 minutes
- Mean evaluations: 165386.6
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
