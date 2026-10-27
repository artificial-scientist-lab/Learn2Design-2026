# Submission rules

## Submission format

Upload a ZIP archive to the [submission portal](https://submit.learn2design2026.com/)
containing the following two mandatory files at its root. Do not place them
inside an enclosing directory.

### `submission.py` — mandatory

This file must define exactly one Python class that subclasses
`dfbench.OptimizationAlgorithm`. The evaluator calls:

```python
MyAlgorithm.optimize(...)
```

### `requirements.txt` — mandatory

List all required Python packages using [PEP 508](https://peps.python.org/pep-0508/)
syntax, with one dependency per line:

```text
<package1>==1.2.3
<package2>==4.5.6
```

### Additional files — optional

You may include other files, such as Python modules or pretrained neural-network
weights, in the archive.

The evaluation script runs from the root of the extracted archive, so your code
can access these files using relative paths.

### Local validation

Run the repository's validator before uploading:

```bash
python tools/validate_submission.py submission.zip
```

By default, this performs static checks only and does not install dependencies
or execute the optimizer. It checks the archive layout, Python syntax,
`OptimizationAlgorithm` subclass, dependency declarations, unsafe archive
entries, and prohibited simulator imports.

The optional local run executes the submission on one newly generated UIFO
topology using the current Python environment and local hardware:

```bash
python tools/validate_submission.py submission.zip \
  --run-local --max-time-seconds 300
```

Omit `--max-time-seconds` to use the official four-hour Objective budget. The
local run reports evaluation and feasibility statistics together with the best
feasible loss. It does not install submission dependencies automatically and
does not reproduce an official leaderboard score. Because `--run-local`
executes code from the archive, use it only with submissions you trust.

Competition entries are not submitted as public pull requests. This keeps
unreleased methods private until the organizers evaluate them, which reduces
copying between participants.

You may submit as often as you want. For each monthly evaluation, the organizers
evaluate only the last submission received before that month's deadline. Earlier
submissions are not evaluated or scored.

---

## Using the Differometor-30k dataset

Submissions are **encouraged** to use the publicly released [`Differometor-30k`](../dataset/README.md)
dataset for pre-training surrogates, initial-point models, warm-starts, or
any other purpose. There is no penalty for spending the entire budget on
surrogate inference while only calling `objective.value()` once. Just keep in
mind that at least one (feasible) logging entry needs to be provided for a run
to be valid.

---

## Evaluation hardware

Each participant run is allocated:

- 1× NVIDIA H100 SXM5 GPU (Hopper, 80 GB HBM3)
- 16 vCPUs on an Intel Xeon Platinum 8468 host
- 128 GiB RAM
- 40 GiB SSD
- Ubuntu 24.04.3 LTS with CUDA 13

Each submission uses a separate Python 3.12.3 environment. See the
[evaluation environment reference](evaluation-environment.md) for the current
NVIDIA driver, fixed core packages, and reference CUDA library versions.

### Batched Objective scaling

The benchmark below shows representative scaling of every batched UIFO
Objective path on one H100. It uses size-3 UIFO problems with 50 frequencies
and reports the mean and SEM across ten topology seeds. The columns show call
latency, throughput relative to batch size 1, and remaining GPU-memory
headroom. The red dotted line marks the first batch size that ran out of GPU
memory.

<p align="left">
  <img src="../media/uifo_batch_scaling_all_operations.png" alt="UIFO batch scaling for all Objective functions" width="720">
</p>

For this representative workload, value-only calls reached batch size 36,
gradient-bearing calls reached 19, and the two Hessian-bearing calls reached 4;
the respective next batch sizes ran out of GPU memory. These are measurements,
not fixed API limits: memory use depends on the topology, problem dimensions,
frequency count, requested outputs, and submission code. Test the exact path
and batch size used by your algorithm against the official resource limits.

---

## Time budget

Official optimization budget per topology: up to 4 hours, measured by
`objective.time_elapsed` after `objective.start_logging()`. Algorithms may
return earlier. Objective-provided JIT warmup before `objective.start_logging()`
does not count.

Additional time allowed for setup, permitted warmup, and result finalization
does not extend the four-hour optimization budget.

Concretely:

- The clock starts when your algorithm calls `objective.start_logging()`.
- Submission computation after that call counts toward the 4 hours, including Objective
  evaluation calls, batched / `vmap_*` variants, optimizer updates, surrogate
  inference, candidate generation, adaptation, and submission-side logging.
- Stop making evaluation calls when `objective.budget_exceeded` becomes `True`.
  Evaluation methods can still perform computation after the budget expires,
  but results outside the budget are not admitted to the scored history.
  The best feasible loss admitted within the budget is what counts.
- Time spent on organizer-managed result recording is excluded from
  `objective.time_elapsed`; submission-side logging counts toward the budget.
- JIT compilation time from Objective-provided `warmup_*()` calls before
  `objective.start_logging()` is not counted against the 4 hours.
- Before calling `objective.start_logging()`, submissions may inspect documented
  public context such as `bounds`, `n_params`, `problem_name`, `problem_spec`,
  and `optimization_pairs`. They may generate random parameters, inspect budget
  configuration, use public pre-run setters, initialize candidates or models
  from that context, and call Objective-provided `warmup_*()` methods.
- Result-producing evaluation methods, raw callable getters such as
  `value_function()`, and `log_evaluation()` are unavailable before logging and
  raise `RuntimeError`. Compiling a custom raw-callable evaluation path therefore
  counts against the 4-hour budget. Accessing private internals or the wrapped
  problem to bypass this lifecycle is forbidden.

---

## Dependencies

Your submission may import any pip-installable Python package, with one exception:

- `import differometor` (and any submodule of it) is forbidden. All
  interaction with the simulator must go through the `Objective` instance you
  are handed.

The evaluation environment ships with a common base (`dfbench`, `differometor`,
`jax`, `jaxlib` CUDA 13, `numpy`, `scipy`, `optax`, `cma`, `cmaes`, `torch`,
`botorch`, `nevergrad`, `evosax`). The evaluator installs the packages listed in
the mandatory root-level `requirements.txt` before running your algorithm.

The simulator and JAX package versions are fixed: declarations for `dfbench`,
`differometor`, `jax`, `jaxlib`, and the JAX CUDA plugins are ignored. Other
requirements, including `nvidia-*` library pins, are honored when compatible
with the fixed core. These NVIDIA libraries are not globally pinned; see the
[evaluation environment reference](evaluation-environment.md#dependency-policy)
for the exact policy and reference versions. Dependency installation finishes
before the timed optimization starts.

---

## Evaluation procedure

1. The evaluator verifies the two mandatory root files and prepares a separate
   Python environment for your submission, installing its dependencies under
   the policy above. The current evaluation setup does not specify a common
   Docker base image.
2. For each run, the archive is extracted into a writable temporary working
   directory. The evaluator runs from that directory and loads the algorithm
   class from `submission.py`. Your algorithm must be able to run without
   network access. Do not rely on files written during one run being available
   in another.
3. For each evaluation, your submission is run on 10 new hidden topologies. For
   each topology, your `optimize()` is called once with a fresh
   [`Objective`](dfbench/Objective-API-Reference.md), a fixed random seed, and
   the budget described above.
4. After the run, the result for that topology is the minimum loss among logged
   setups with `is_feasible=True`.
5. If your run has no feasible setup, that topology result is replaced by the
   best feasible loss from the organizers' random-search baseline on the same
   topology.
6. Your score for the month is the arithmetic mean of the 10 topology results.

---

## Prize eligibility and source disclosure

Submissions ranking in the top 10 of the final private leaderboard must provide
full source code to the organizers for manual review. Code need not be made
public, but the organizers must be able to verify that no banned operations were
used. Prize money is withheld until review is complete.

---

## Disqualification criteria

- Encoding knowledge of private topology specifications in any form.
- Bypassing the `Objective` limitations to evaluate, obtain raw evaluation
  callables early, or record results before `objective.start_logging()`.
- Submitting work that is not your own without attribution.
- Violations of the [Code of Conduct](../CODE_OF_CONDUCT.md).
