# Learn2Design 2026 Round 2 evaluation data

This public release provides detailed evaluation statistics for all 60 Round 2
participants. It includes the ten run scores, aggregate uncertainty, efficiency
and feasibility statistics, and selected convergence checkpoints.

## Contents

- `leaderboard.csv`: aggregate score and run statistics for every participant.
- `runs.csv`: exact final statistics for all 600 participant runs.
- `checkpoints.csv`: per-run values at 15, 30, 60, 120, 180 and 240 minutes.
- `checkpoint_summary.csv`: checkpoint mean, sample standard deviation and SEM.
- `seeds.csv`: the shared Round 2 topology and optimizer seed pairs.
- `participants/<participant_id>/`: an individual report and convergence plot.

## Shared seeds

All participants and organizer baselines used the same ten seed pairs. Round 2
public-evaluation topologies are retired; later rounds and the final evaluation
use different hidden topologies.

| Run | Seed index | Topology seed | Optimizer seed |
|---:|---:|---:|---:|
| 1 | 0 | 289505736 | 585204139 |
| 2 | 1 | 1193264097 | 995044121 |
| 3 | 2 | 1771871117 | 1434201994 |
| 4 | 3 | 124263112 | 1945652744 |
| 5 | 4 | 4814052 | 493480448 |
| 6 | 5 | 1111423186 | 125027229 |
| 7 | 6 | 1166557331 | 1153707073 |
| 8 | 7 | 2108583176 | 550034327 |
| 9 | 8 | 811189383 | 1418116971 |
| 10 | 9 | 268784919 | 766688992 |

Lower loss is better. Standard deviations use the sample definition (`ddof=1`),
and SEM is `std / sqrt(10)` for complete final results. Intermediate checkpoints
are sampled upper bounds unless marked as exact; the 240-minute values are the
exact official run scores.
