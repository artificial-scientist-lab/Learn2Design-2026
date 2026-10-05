# doris_south — Round 3 evaluation data

## Summary

- Place: 66 of 67
- Mean feasible loss: 4.311330066
- Sample standard deviation: 0.599392966
- Standard error of the mean: 0.189544699
- Median feasible loss: 4.363505090
- Minimum / maximum run score: 2.964400668 / 5.106120441
- Mean time to best: 41.69 minutes
- Mean evaluations: 11817.4
- Failed runs: 0
- Random-search fallback runs: 1

## Files

- `summary.json`: machine-readable aggregate statistics.
- `runs.csv`: exact final outcome and efficiency statistics for all ten runs.
- `checkpoints.csv`: per-run best feasible loss at selected elapsed times.
- `convergence.csv`: mean and uncertainty across runs at those checkpoints.
- `convergence.png`: visual summary of the convergence data.

Intermediate checkpoints are conservative upper bounds unless marked exact.
The 240-minute values are official run scores; missing values and fallbacks
are marked.

Run 3 found no feasible candidate; its score uses RandomSearch on the same seed
pair. Its runtime and evaluation count are unchanged.
