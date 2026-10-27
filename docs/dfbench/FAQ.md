# FAQ and Troubleshooting

---

## Algorithm Implementation

### What objective function is optimized for UIFO?

For `UIFOProblem`, your algorithm evaluates `objective.value(params)`, which returns one scalar loss:

```text
loss = sensitivity_loss + penalty
```

The `sensitivity_loss` compares the candidate detector's broadband strain-sensitivity curve to the Voyager reference detector on a log scale. Lower is better. Roughly, `0` means reference-level sensitivity, negative values mean better-than-reference sensitivity, and positive values mean worse sensitivity.

The `penalty` comes from physical power constraints. During each evaluation, the simulator computes optical powers at constrained component groups and applies the active `power_penalty_fn(value, threshold)` wherever a threshold is exceeded. The default penalty is a squashed ReLU-style penalty, but constrained problems let you replace it before logging starts with `objective.set_penalty_fn(...)`.

For scoring, feasibility still matters separately from the numeric penalty: each topology result is the minimum loss among setups with `is_feasible=True`, and the leaderboard score is the arithmetic mean over 10 new hidden topologies. If a run has no feasible setup, the organizers use the best feasible loss from random search on that topology as the replacement result. Use aux methods such as `objective.value_aux(params)` or save tokens such as `is_feasible`, `sensitivity_loss`, and `penalty` to inspect the loss decomposition. To build intuition, open the [interactive sensitivity loss explorer](../sensitivity_loss_explorer.html). See [Power Constraints and Aux Diagnostics](../dfbench_overview.md#power-constraints-and-aux-diagnostics) and [`UIFOProblem`](Problems.md#uifoproblem) for details.

### My gradient method diverges to NaN

1. **Learning rate too high.** Differometor loss landscapes are steep. Start with `lr=0.01` or lower.
2. **Box constraints causing zero gradients.** If parameters hit bound edges frequently, gradients become zero. Consider using `unbounded=True` so sigmoid bounding provides smooth gradients everywhere.
3. **Using `obj.grad()` instead of `obj.value_and_grad()`.** `grad()` does **not** log a loss: use `value_and_grad()` to get both.

### Bounded vs. unbounded: which do I use?

| Algorithm type | Typical choice | Why |
|----------------|----------------|-----|
| Evolutionary | `False` | Populations naturally respect bound constraints. |
| Derivative-free | `False` | Direct-search and Powell-style solvers usually operate on native box bounds. |
| Global search | `False` | SciPy global optimizers sample and perturb candidates in bounded physical space. |
| Surrogate-based | `False` | GP/BO acquisitions work in bounded space. |
| Gradient-based | Either | Use `True` if you want smooth unconstrained space for gradient flow. Use `False` if your method handles box constraints directly. |
| Generative | Either | Depends on internal representation. |

**Note:** The `Benchmark` harness defaults gradient-based algorithms to `unbounded=True` for convenience, but this can be overridden if your algorithm works in bounded space; see the [bounded/unbounded guide](Implementing-a-New-Algorithm.md#bounded-vs-unbounded-detailed-guide).

### How do I convert between bounded and unbounded parameters?

```python
from dfbench import inverse_sigmoid_bounding

# bounded -> unbounded
unbounded_params = inverse_sigmoid_bounding(bounded_params, problem.bounds)

# unbounded -> bounded is done automatically during evaluation when obj.unbounded=True
```

### My algorithm uses PyTorch

```python
from dfbench import t2j, j2t

params_jax = t2j(params_torch)       # PyTorch -> JAX
losses = obj.vmap_value(params_jax)
losses_torch = j2t(losses)            # JAX -> PyTorch
```

The conversion goes through NumPy and adds negligible overhead; see [tensor conversion helpers](Utilities-and-Helpers.md#tensor-conversion-t2j--j2t).

---

## Benchmarking

### How many runs should I use?

At least 30 for basic statistics, 100+ for reliable confidence intervals. The default is 100.

### Can I re-evaluate metrics without re-running?

Yes. Save run data with `save_run_data=True`, then reload with different settings; see [Benchmark orchestration](Architecture-Overview.md#benchmark-orchestration):

```python
benchmark = Benchmark(problem, success_loss=0.05, ..., n_runs=100, max_time=300)
results = benchmark.run(load_from="./data/benchmark_run_data/VoyagerProblem_...")
```

### What does `random_baseline_loss` do?

Enables **normalized AUC**: each run's AUC is divided by the AUC of a hypothetical constant-loss algorithm, then log-scaled. Set it to the expected loss from random parameter sampling.

---

## HPC and Environment

### matplotlib crashes with `PermissionError`

dfbench auto-redirects `MPLCONFIGDIR` to a temp directory at import time. Ensure you import dfbench before matplotlib:

```python
import dfbench        # sets MPLCONFIGDIR
import matplotlib     # now safe
```

### JAX uses too much GPU memory

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

This makes JAX allocate memory on demand instead of pre-allocating 75% of GPU memory.

### Multiple jobs fail on the same GPU node

Use SLURM's `--gres=gpu:1` to isolate GPU access, or set `CUDA_VISIBLE_DEVICES` to assign specific GPUs.

---

## Data and I/O

### Where are results saved?

| Data type | Default path |
|-----------|-------------|
| Benchmark CSV | `./data/benchmark_results/` |
| Run data (NPZ) | `./data/benchmark_run_data/` |
| Objective run data | `./data/objective_run_data/` |
| Problem outputs | `./data/problem_output/` |

### `Objective.save_run_data()` vs. benchmark saving

`Objective.save_run_data()` saves a single run (for development/debugging). The benchmark's `save_run_data` flag saves all runs for all algorithms in a structured directory with metadata for later re-evaluation.

### Legacy data format

The benchmark loader auto-detects and converts legacy files that used `all_wall_time_indices` instead of per-evaluation `time_steps`.
