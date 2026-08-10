# Contributing

Learn2Design-2026 is the competition repository. It holds the starting kit, baselines, the precomputed dataset, and the rules. The framework that runs your algorithm (`dfbench`) lives in a separate repo.

By participating you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Submitting an algorithm

Submission ZIP archives go through the [competition submission portal](https://submit.learn2design2026.com/),
not through pull requests on this repo. Read
[`docs/submission.md`](docs/submission.md) for the full rules (archive layout,
time budget, dependencies, and disqualification criteria) and
[`docs/scoring.md`](docs/scoring.md) for how your run is scored.

## Improving the framework, baselines, or docs

The `dfbench` benchmark framework, the algorithm implementations, and the reference docs are developed in the [Differometor-Benchmark](https://github.com/artificial-scientist-lab/Differometor-Benchmark) repository. Bug reports, fixes, new reference algorithms, and doc improvements all belong there, not here.

To contribute:

1. Open an issue in [Differometor-Benchmark](https://github.com/artificial-scientist-lab/Differometor-Benchmark/issues) describing the change for anything beyond a trivial fix.
2. Open a PR against `main` in [Differometor-Benchmark](https://github.com/artificial-scientist-lab/Differometor-Benchmark).
3. Add a line to `CHANGELOG.md` under `## [Unreleased]`.

## What belongs in this repo

Pull requests against Learn2Design-2026 are welcome for:

- Fixes to the README, the competition docs in `docs/`, or the dataset guide in `dataset/`.
- Corrections to the bundled `dfbench` doc snapshot in `docs/dfbench/` (the source of truth is in Differometor-Benchmark, so non-trivial changes should go there and be re-synced here).

We do not merge competition submissions, new baselines, or framework changes into this repo. Those go through the portal or through Differometor-Benchmark respectively.

## Security

If you find a security issue, do **not** open a public issue. Email `jonathan.klimesch@uni-tuebingen.de` and `mario.krenn@uni-tuebingen.de` directly. We will respond rapidly.
