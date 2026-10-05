# Learn2Design 2026 Round 3 evaluation data

This release provides evaluation statistics for the 67 participants on the
Round 3 leaderboard: ten run scores, uncertainty, efficiency and feasibility
statistics, and convergence checkpoints.

## Contents

- `leaderboard.csv`: aggregate score and run statistics for every participant.
- `runs.csv`: exact final statistics for all 670 participant runs.
- `checkpoints.csv`: per-run values at 15, 30, 60, 120, 180 and 240 minutes.
- `checkpoint_summary.csv`: checkpoint mean, sample standard deviation and SEM.
- `seeds.csv`: the shared Round 3 topology and optimizer seed pairs.
- `participants/<participant_id>/`: an individual report and convergence plot.

## Shared seeds

All participants and organizer baselines used the same ten seed pairs. Round 3
public-evaluation topologies are retired; later rounds and the final evaluation
use different hidden topologies.

| Run | Seed index | Topology seed | Optimizer seed |
|---:|---:|---:|---:|
| 1 | 0 | 98591154 | 1791403674 |
| 2 | 1 | 650941223 | 887131975 |
| 3 | 2 | 1211049875 | 1078728795 |
| 4 | 3 | 1363760596 | 1142570359 |
| 5 | 4 | 1795982962 | 398006495 |
| 6 | 5 | 196619468 | 773385860 |
| 7 | 6 | 239637960 | 1603749681 |
| 8 | 7 | 1410548495 | 872722144 |
| 9 | 8 | 791199028 | 1571924013 |
| 10 | 9 | 1082220220 | 687175312 |

Lower loss is better. Standard deviations use the sample definition (`ddof=1`),
and SEM is `std / sqrt(10)` for complete final results. Intermediate checkpoints
are conservative upper bounds unless marked exact; the 240-minute values are
the official run scores. Missing values are left blank, and fallbacks are marked.

Feasibility fractions refer to history observations, not all evaluated candidates.
Mean time to best includes only runs with a feasible result.
