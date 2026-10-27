# Round 4 top-ten comparison on 50 UIFOs

Evaluation statistics for the ten leading teams on fifty shared UIFOs:
the ten official UIFOs plus forty additional UIFOs. Scores and ranks apply
to this comparison; the [official leaderboard](../) uses ten UIFOs per team.

## Contents

- `leaderboard.csv`: aggregate score and run statistics for every participant.
- `runs.csv`: final outcome, timing, evaluation counts and feasibility statistics.
- `checkpoints.csv`: per-run values at 15, 30, 60, 120, 180 and 240 minutes.
- `checkpoint_summary.csv`: checkpoint mean, sample standard deviation and SEM.
- `seeds.csv`: the shared topology and optimizer seed pairs.
- `participants/<participant_id>/`: an individual report and convergence plot.

Lower loss is better. Standard deviations use the sample definition (`ddof=1`),
and SEM is `std / sqrt(n)` for the available runs. Intermediate checkpoints
are conservative upper bounds unless marked exact; the 240-minute values
are final run scores. Missing values are left blank, and fallbacks are marked.

Feasibility fractions refer to history observations, not all evaluated candidates.
Mean time to best includes only runs with a known best-result time.
