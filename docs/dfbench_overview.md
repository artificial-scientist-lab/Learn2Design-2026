# Differometor Benchmark - [dfbench](https://github.com/artificial-scientist-lab/Differometor-Benchmark)

## What `Objective` gives you

A quick summary of the methods you'll touch most often. Data logging is done automatically (you will have to choose which metrics to log by [`save`](dfbench/Objective-API-Reference.md#choosing-what-to-save) parameter). Budget is tracked and enforced after each call.

### Evaluation (single-point)

| Method | Purpose |
|---|---|
| `objective.value(params)` | Forward pass → scalar loss. |
| `objective.grad(params)` | Gradient only (no loss computed). |
| `objective.hessian(params)` | Exact Hessian. |
| `objective.value_and_grad(params)` | Loss + gradient in one forward+backward pass (preferred over separate calls). |
| `objective.value_grad_and_hessian(params)` | Loss + gradient + Hessian together. |

### Evaluation (batched via `jax.vmap`)

| Method | Purpose |
|---|---|
| `objective.vmap_value(params_batch)` | Batched losses, shape `(batch,)`. Counts as `batch` evals. |
| `objective.vmap_value_and_grad(params_batch)` | Batched `(losses, grads)`. |
| `objective.vmap_value_grad_and_hessian(params_batch)` | Batched `(losses, grads, Hessians)`. |
| `objective.batched_value(…)`, `batched_grad(…)`, … | Aliases for the `vmap_*` methods above. |

### Lifecycle & setup

| Method / attribute | Purpose |
|---|---|
| `objective.warmup_value()`, `warmup_value_and_grad()`, `warmup_vmap_*`, … | JIT-compile before the timer starts (free, not budgeted). Call before `start_logging()`. |
| `objective.start_logging()` | Start the official 4-hour wall-clock timer; call after warmup. |
| `objective.set_space_mode(unbounded, unit_mapping=None, inverse_unit_mapping=None)` | Switch bounded/unbounded space (and optionally the [0,1] mapping pair) before optimization. |
| `objective.set_seed(seed)` | Seed the internal PRNG for reproducible `random_params*` draws. |
| `objective.set_penalty_fn(fn)` | Swap the penalty function on constrained problems (`UIFOProblem`, …); retraces JIT paths. Use `relu_penalty`, `zero_penalty`, etc. |

### Evaluation (aux — constrained problems only)

| Method | Purpose |
|---|---|
| `objective.value_aux(params)` | `(loss, aux)` dict: `sensitivity_loss`, `penalty`, `is_feasible`, `violations`, `power_values`. |
| `objective.value_and_grad_aux(params)` | Loss + gradient + aux in one pass. |
| `objective.vmap_value_aux(...)`, `vmap_value_and_grad_aux(...)` | Batched aux variants. |

### Sampling

| Method | Purpose |
|---|---|
| `objective.random_params(n_samples=1)` | Sample from the *active* space (bounded if `unbounded=False`, else unbounded). |
| `objective.random_params_bounded(n_samples=1)` | Uniform samples inside `problem.bounds`. |
| `objective.random_params_unbounded(n_samples=1)` | Bounded samples mapped to unbounded space via the active inverse mapping. |

### Budget & state

| Attribute | Purpose |
|---|---|
| `objective.budget_exceeded` | Main loop-termination check (time **or** evals exhausted). |
| `objective.budget_left_fraction`, `budget_progress_fraction` | Fraction of the tightest active budget remaining / consumed. |
| `objective.eval_count`, `evals_left`, `evals_exceeded` | Evaluation-counter view. |
| `objective.time_elapsed`, `time_left`, `time_exceeded` | Wall-clock view (relative to `start_logging()`). |
| `objective.evals_since_improvement`, `improvement_count` | For patience-based early stopping. |

### Results & history

| Attribute | Purpose |
|---|---|
| `objective.best_loss` | Lowest raw loss observed (`None` before the first eval); not necessarily the scoring value. |
| `objective.best_params_bounded` | Best params mapped to bounded space. Use this for final output. |
| `objective.best_params`, `best_eval_index`, `best_batch_index` | Best params (raw space), and where in history it lives. |
| `objective.best_is_feasible` | Physical feasibility of the best-loss point (needs `is_feasible` save token; constrained problems). |
| `objective.loss_history`, `params_history`, `time_steps` | Raw recorded histories (copies). |
| `objective.*_history_reduced` | Batched histories collapsed to one representative entry (lowest loss) per step. |
| `objective.sensitivity_loss_history`, `penalty_history`, `is_feasible_history`, `violations_history`, `power_*_history` | Aux diagnostics (enable via `save=[...]` tokens on constrained problems). |

### Problem & configuration

| Attribute | Purpose |
|---|---|
| `objective.bounds` | Per-parameter `(low, high)` arrays (`±inf` when unbounded). |
| `objective.n_params` | Number of optimizable parameters. |
| `objective.problem` | The wrapped `ContinuousProblem`. |
| `objective.penalty_fn` | Active penalty callable (or `None`). |
| `objective.power_thresholds` | Per-group physical thresholds `{hard, soft, detector}` for constrained problems, or `None`. |

### Custom JIT loops

| Method | Purpose |
|---|---|
| `objective.value_function(unbounded=None)` | Unlogged pure JAX callable for use inside your own JIT loop. |
| `objective.log_evaluation(params=…, loss=…, grad=…, hessian=…)` | Manually record a completed step when using `value_function()`. |

### Persistence

| Method | Purpose |
|---|---|
| `objective.save_run_data(...)` / `load_run_data(filepath)` | Atomic checkpoint save/load; `time_elapsed` continues seamlessly across resume. |
| `objective.output_to_files(...)` | Human-readable JSON + PNG loss/sensitivity plots. |
| `objective.get_summary()` | Snapshot dict of eval count, time, best/current loss, budget flags. |

Full reference: [local `dfbench` docs](dfbench/Objective-API-Reference.md) or [`dfbench` wiki](https://github.com/artificial-scientist-lab/Differometor-Benchmark/wiki)

---

## Explanation of the search space

### Bounds

In this competition, you are optimizing an objective function `objecitve.value(params)`
that (in the case of UIFO) has ~200 parameters depending on its topology (represented as a JAX Array `params`). Each parameter is by
default bounded to a space that depends on the functionality of the corresponding component:

| Property | Bounds | Components with property |
| --- | --- | --- |
| `reflectivity` | `[0, 1]` | `mirror`, `beamsplitter` (interior grid mirrors `ml`/`mr`/`mt`/`mb` and cell-center beamsplitters) |
| `tuning` | `[-360, 360]` | `mirror`, `beamsplitter` |
| `mass` | `[0.01, 200]` | `free_mass` (suspensions of the interior mirrors and of the cell-center beamsplitters) |
| `length` | `[0.1, 4000]` | `space` (edges connecting components; parallel inter-cell lengths are tied together by `constrain_inter_grid_cell_spaces` to preserve the grid structure, so each tied group shares one optimization variable) |
| `power` | `[0, 200]` | `laser` (boundary lasers; the balanced-homodyne local oscillator is held fixed via `not_optimizable=True`) |
| `db` | `[0, 10]` | `squeezer` (boundary squeezers) |
| `angle` | `[-360, 360]` | `squeezer` (boundary squeezers) |

The bounds above are the UIFO defaults. You can retrieve the exact number of params and bounds of the problem inside objective by running

```python
objective.n_params  # int
objective.bounds  # -> Array of [lower_bound_array, upper_bound_array]
```

where `bounds` is a `2 x n_params` Array. `bounds[0]` is the minimal parameter vector; `bounds[1]` the maximal.

### Unbounded Parameter Space

The parameter space can be unbounded such that any real-valued vector is valid. You can either use the `dfbench` default sigmoid transformation or choose your own. Unbounding is useful for gradient-based optimizers that expect an unbounded search space (-∞, +∞). The Objective maps your unbounded parameters back into the problem bounds before evaluating the physics.

**You can either set this at construction:**

```python
from dfbench import Objective

# DEFAULT sigmoid mapping
objective = Objective(problem, unbounded=True)

# CUSTOM (forward maps to [0, 1], Objective scales to actual bounds)
import jax
objective = Objective(
    problem,
    unbounded=True,
    unit_mapping=jax.nn.sigmoid,
    inverse_unit_mapping=lambda x: jax.numpy.log(x / (1.0 - x)),  # sigmoid inverse
)
```

**Or before logging:**

```python
objective = Objective(problem)  # bounded by default
# before start_logging():
# DEFAULT
objective.set_space_mode(True)  # sigmoid

# CUSTOM
objective.set_space_mode(
    True,
    unit_mapping=jax.nn.sigmoid,
    inverse_unit_mapping=lambda x: jax.numpy.log(x / (1.0 - x)),
)

...

objective.start_logging()
```

Rules:

- `set_space_mode` must be called before `start_logging()`.
- If you pass a custom mapping `unit_mapping`, you must also pass `inverse_unit_mapping`. The forward mapping must produce values in [0, 1]. The Objective handles scaling to actual bounds via `bounded = lb + (ub - lb) * f(x)`. The inverse receives values already normalised to [0, 1]: `unbounded = f_inv((bounded - lb) / (ub - lb))`.
- Both callables can be scalar functions (e.g. `jax.nn.sigmoid`) or element-wise vector functions — JAX broadcasts element-wise operations, so both work.
- The default pair is sigmoid + inverse-sigmoid (logit). You are not required to handle bounds scaling yourself.

When unbounded, `objective.best_params` is in unbounded space; use `objective.best_params_bounded` (and `objective.params_history_bounded`) to recover the physically meaningful bounded parameters for output.

---

## Explanation of "Topology"
### What is a topology?

A topology fixes which optical components are placed where, and how they are wired together with spaces. You do not get to change the topology during optimization. You only optimize the continuous parameters attached to it (reflectivities, tunings, lengths, masses, powers, squeezing).

The UIFO is built on a grid. For `size=3` (used in the competition) you get a 3x3 interior of unit cells, each one a beamsplitter-like element in the center surrounded by four mirrors. Around that interior sits a ring of boundary cells, each holding a source or a readout.

![UIFO topology](../media/UIFO.png)

The figure shows the UIFO grid. The components you can swap when picking a topology are color-coded: the center of each interior cell can be either a `beamsplitter` or a `directional_beamsplitter`, in one of four orientations (left/right/top/bottom), and each boundary cell can be a `laser`, `squeezer`, `detector`, or `balanced_homodyne`. Everything else (the surrounding mirrors, the suspensions, the spaces, the signal nodes) is implied by the grid structure and is the same across topologies of the same size.

### Setting the topology

There are three ways to pin down a topology, and they are mutually exclusive:

1. `topology_seed` is the lazy option. A topology is sampled deterministically from the seed, and the seed is printed to the console so you can reproduce it.

   ```python
   from dfbench.problems import UIFOProblem
   problem = UIFOProblem(size=3, topology_seed=42)
   ```

2. `topology` is a compact string with meaning. Interior cells use `A`-`H`, boundaries use `L`/`S`/`D`/`H`, separated by a dash:

   | Code | Meaning |
   | --- | --- |
   | `A`-`D` | `beamsplitter`, orientation left/right/top/bottom |
   | `E`-`H` | `directional_beamsplitter`, orientation left/right/top/bottom |
   | `L` | `laser` |
   | `S` | `squeezer` |
   | `D` | `detector` |
   | `H` | `balanced_homodyne` |

   Interior chars are row-major over the `size x size` grid. Boundary chars scan the ring (top row, then left and right columns, then bottom row), skipping corners. For `size=3` that's 9 interior + 12 boundary = 21 characters.

   ```python
   problem = UIFOProblem(size=3, topology="AECGCCHEG-SLLSSHLLLLS")
   ```

3. `centers` + `boundaries` dicts give you full control, in Differometor's native format (refer to [`uifo_problem.py`](https://github.com/artificial-scientist-lab/Differometor-Benchmark/blob/main/src/dfbench/problems/uifo/uifo_problem.py)):

   ```python
   problem = UIFOProblem(
       size=3,
       centers={"11": ("beamsplitter", "left"), "12": ("directional_beamsplitter", "top")},
       boundaries={"01": "squeezer", "14": "detector"},
   )
   ```

You can convert between the dict and string forms with `topology_to_string` and `topology_from_string` from `dfbench.problems.uifo`.

A note on the grid: horizontal and vertical inter-cell spaces at the same grid position are tied to the same length parameter. That keeps the grid from folding into a degenerate geometry during optimization. The cells stay aligned in a grid position with different distances in between.

---

## What to train/test your algorithm on?

The problem you are scored on is `UIFOProblem` which has ~200 parameters (depending on topology). Each evaluation uses 10 new hidden topologies. Official budget per topology is exactly 4 hours of wall-clock time after `objective.start_logging()`. On an A100 a single evaluation takes roughly 500 ms once JIT is warm.

For development there is a smaller problem that uses the same loss computation: `ConstrainedVoyagerProblem`. It is the aLIGO Voyager design, roughly 25 components, and it uses the same three noise sources and the same power-constraint penalty (i.e. the same loss function) as the UIFO. There are just fewer components. It takes roughly 20x less time per evaluation than the UIFO on the same hardware (around 25 ms/eval on an A100). You can use it to speed up the evaluation loop and get a better feel for your algorithm.

![Voyager layout](../media/voyager.png)

The figure shows the Voyager components. You can think of the UIFO as a grid-shaped generalization of Voyager. From `ConstrainedVoyagerProblem`, the only thing that changes when you move to `UIFOProblem` is the parameter count and the per-eval cost; the Objective, the penalty contract, the aux diagnostics, and the loss semantics are identical. This of course changes the the loss-landscape significantly but `ConstrainedVoyagerProblem` can nevertheless give you a feel for your algorithm:

```python
from dfbench.problems import ConstrainedVoyagerProblem
from dfbench import Objective

problem = ConstrainedVoyagerProblem()  # No topology!
objective = Objective(problem, max_time=300)  # Depending on hardware, 5 minutes can already be enough to converge
```

One more difference: `UIFOProblem` topologies can be randomized, while `ConstrainedVoyagerProblem` is a single fixed design. So Voyager is good for mechanics and hyperparameter sweeps, but it does not tell you how your algorithm generalizes across topologies or even the UIFO in general. For that you need to run on `UIFOProblem` with several different `topology_seed` values. Iterate through different seeds there.

There are two more problems in dfbench (`VoyagerProblem`, `VoyagerTuningProblem`) that use a simpler single-noise model and have no power constraints. They are useful for unit tests and quick sanity checks but are not representative of the competition objective. See [docs/dfbench/Problems.md](dfbench/Problems.md) for the full hierarchy.

---

## Power Constraints and Aux Diagnostics

The UIFO loss is not just composed of points on the sensitivity curve. It also penalizes optical power that exceeds physical thresholds, and it exposes a set of auxiliary diagnostics you can read alongside the loss. This matters for how you score and what you log.

For a browser-based intuition builder, open the [interactive sensitivity loss explorer](sensitivity_loss_explorer.html).

The optimization is subject to physical feasibility. This corresponds to the boolean `is_feasible` below. For scoring, only the minimum loss among feasible setups in each run counts. If a run contains no feasible setup, the organizers replace that topology result with the best feasible loss found by random search on the same topology. You are allowed, and encouraged, to modify the penalty function to fit your algorithm best, but the resulting loss is only relevant to the score inside the feasible regime.

### The penalty term

Three power thresholds are enforced, one per component group:

| Group | Threshold constant | What it limits |
| --- | --- | --- |
| `hard` | `HARD_SIDE_POWER_THRESHOLD` (3.5e6 W) | Power on mirror/beamsplitter ports with coating |
| `soft` | `SOFT_SIDE_POWER_THRESHOLD` (2e3 W) | Power on mirror/beamsplitter ports without coating |
| `detector` | `DETECTOR_POWER_THRESHOLD` (1e-2 W) | Power on detector ports |

For each evaluation the problem computes the per-group powers, calls `power_penalty_fn(value, threshold)` element-wise, and sums the results into the loss. Three presets ship with dfbench:

| Preset | Formula | Import |
| --- | --- | --- |
| `squashed_relu_penalty` (default) | `max(v/t - 1, 0) / (1 + max(v/t - 1, 0))` | `from dfbench.problems import squashed_relu_penalty` |
| `relu_penalty` | `max(v/t - 1, 0)` | `from dfbench.problems import relu_penalty` |
| `zero_penalty` | `0` | `from dfbench.problems import zero_penalty` |

The default is squashed because a raw ReLU penalty can be orders of magnitude larger than the sensitivity loss. The squashed version bounds the penalty contribution while keeping the gradient direction. Other penalty functions may work better for some algorithms. You can pass any callable with signature `fn(value: float, threshold: float) -> float (penalty)`. This functino must be positive (≥0) across all values!

You can swap the penalty after the problem is constructed but before logging starts, through the Objective:

```python
from dfbench import Objective
from dfbench.problems import ConstrainedVoyagerProblem, zero_penalty

problem = ConstrainedVoyagerProblem()
objective = Objective(problem)
objective.set_penalty_fn(zero_penalty)   # turn the penalty off
objective.warmup_value()
objective.start_logging()
```

Only problems that opt into the power-penalty contract (`ConstrainedVoyagerProblem`, `UIFOProblem`) accept `set_penalty_fn`. The only problems you should use for this competition.

### The aux dict

On the same problems, `objective_function_aux(params)` runs the same forward pass as `objective_function(params)` but also returns a pytree dict with the loss decomposition and physical diagnostics. You can access this through these methods:

```python
loss, aux = obj.value_aux(params)                      # (float, dict)
loss, grad, aux = obj.value_and_grad_aux(params)       # (float, Array, dict)
losses, aux = obj.vmap_value_aux(params_batch)         # (Array[batch], dict with batches)
losses, grads, aux = obj.vmap_value_and_grad_aux(params_batch)
```

The dict has this shape:

| Key | Shape | Description |
| --- | --- | --- |
| `sensitivity_loss` | scalar | The unpenalised sensitivity loss. This is what you would get with `zero_penalty`. |
| `penalty` | scalar | The summed penalty contribution at this point. |
| `is_feasible` | scalar bool | `True` iff every per-group power is at or below its threshold. This is a physical check, independent of the active `power_penalty_fn`, so it stays meaningful even when the penalty is disabled with `zero_penalty`. |
| `violations` | `(n_constraints,)` | Per-constraint penalty values. |
| `power_values` | dict with `hard`, `soft`, `detector` leaves | Raw per-group power arrays. |

Because `aux` is a JAX pytree, the batched variants add a leading batch dim to every leaf, including the `power_values` sub-arrays. The gradient returned by `value_and_grad_aux` is taken with respect to the loss (including penalty).

### Auto-logging aux from the standard methods

You do not have to call the `*_aux` methods to get aux data into your histories. If you enable at least one aux save token via:

```python
problem = UIFOProblem()
Objective(
    problem,
    max_time=600,
    save=["loss", "is_feasible"]  # is_feasible is an aux token, see above
)
```

...the standard loss-bearing methods (`value`, `value_and_grad`, `vmap_value`, `vmap_value_and_grad`, `value_grad_and_hessian`, `vmap_value_grad_and_hessian`) run the aux objective in the same forward pass and record the enabled aux fields alongside the loss. There is no second forward pass and the return signatures do not change; `value` still returns a scalar, `value_and_grad` still returns `(loss, grad)`. The aux pytree is stashed internally and fed to the histories.

Grad-only and Hessian-only calls do not compute a loss, so they have no aux to record. They append `None` placeholders to the enabled aux histories so those stay length-aligned with `loss_history`.

Auto-logging is on iff both hold: at least one aux token is in the `save` list, and the problem opts into the penalty contract.

### Choosing what to save

The Objective always records losses. Two boolean flags toggle the common histories:

| Flag | Default | Effect |
| --- | --- | --- |
| `save_time_steps` | `True` | Record elapsed-time stamps per evaluation |
| `save_params_history` | `True` | Record parameter vectors (reduced for batches) |
| `save_batched_params_history` | `False` | Store full `(batch, n_params)` parameter arrays instead of the reduced representative point |

Beyond that, pass a list of string tokens to `save`. Each token controls one history, so you can enable `is_feasible` without also storing the bulky `power_values` arrays.

Standard tokens:

| Token | Effect |
| --- | --- |
| `grad` | Gradient history (reduced to one entry per eval for batches) |
| `hessian` | Hessian history (reduced for batches) |
| `eval_type` | Per-eval type bitmask history |
| `batched_loss` | Full `(batch,)` loss vectors instead of batch min |
| `batched_grad` | Full `(batch, n_params)` gradient arrays |
| `batched_hessian` | Full `(batch, n_params, n_params)` Hessian arrays |
| `batched` | Convenience alias for the three `batched_*` tokens above |

Aux tokens:

| Token | Effect |
| --- | --- |
| `sensitivity_loss` | Per-eval unpenalised sensitivity loss |
| `penalty` | Per-eval summed penalty |
| `is_feasible` | Per-eval physical feasibility flag |
| `power_values` | Per-eval per-group powers (hard, soft, detector) |
| `violations` | Per-eval per-constraint violation arrays |
| `aux` | Convenience alias for the five non-batched aux tokens above |
| `batched_sensitivity_loss` | Full batched sensitivity loss arrays |
| `batched_penalty` | Full batched penalty arrays |
| `batched_is_feasible` | Full batched feasibility bool arrays |
| `batched_power_values` | Full batched per-group power arrays |
| `batched_violations` | Full batched per-constraint violation arrays |
| `batched_aux` | Convenience alias for the five `batched_*` aux tokens above |

When a `batched_*` aux token is off and the corresponding non-batched token is on, batched aux entries are reduced to the representative point (the index of the best loss within the batch), so the recorded `is_feasible` and `violations` reflect that best point. This matches the reduction rule used for gradients and Hessians.

```python
# gradients, full batched losses, and feasibility
obj = Objective(problem, save=["grad", "batched_loss", "is_feasible"])

# everything aux, with full batched feasibility arrays
obj = Objective(problem, save=["aux", "batched_is_feasible"])
```

### Reading aux back

The aux histories are exposed as properties on the Objective, one per field. They return copies and stay aligned with `loss_history` by index:

```python
obj.is_feasible_history       # list, one entry per logged eval (or None placeholder)
obj.sensitivity_loss_history  # list
obj.penalty_history           # list
obj.violations_history        # list of (n_constraints,) arrays
obj.power_hard_history        # list of per-eval hard-group power arrays
obj.power_soft_history        # list of per-eval soft-group power arrays
obj.power_detector_history    # list of per-eval detector-group power arrays
```

The thresholds themselves are exposed as a read-only property, so you can interpret the `power_*` histories without hardcoding constants:

```python
obj.power_thresholds     # {"hard": 3.5e6, "soft": 2e3, "detector": 1e-2}, or None
```

It returns `None` on problems that do not opt into the penalty contract. The thresholds are physical constants; they do not change across evaluations or after `set_penalty_fn`.

For the best-loss point specifically:

```python
obj.best_loss            # the lowest loss observed
obj.best_params_bounded  # the params that produced it, in bounded space
obj.best_is_feasible     # True/False/None (None if is_feasible was never enabled)
```

`best_is_feasible` returns `None` when the `is_feasible` token was never enabled, when no evaluation has improved yet, or when the best point came from a non-aux call. For a batched best loss it returns the feasibility of the winning batch element when `batched_is_feasible` is on, otherwise the per-call reduced entry.

### Use during evaluation

For the competition, we will log `is_feasible` together with the loss and parameters during evaluation. If the algorithm is using `vmap`, all three tokens are batched to preserve information about all points evaluated. The raw `objective.best_loss` does not need to be feasible; scoring uses the best logged feasible setup, with the random-search replacement rule if none exists.
