# Objective API Reference

`Objective` is the central class of the benchmark. It wraps a `ContinuousProblem` and acts as the only interface between an optimization algorithm and the underlying physics simulation. Every function evaluation, gradient computation, and random sample goes through `Objective`, which transparently records everything needed for reproducible benchmarking.

For rare cases that need a raw JAX-compatible callable inside a custom JIT loop, start logging, use `Objective.value_function(...)`, and then record completed evaluations with `Objective.log_evaluation(...)`.

**Import:**

```python
from dfbench import Objective
```

---

## Constructor

```python
Objective(
    problem: ContinuousProblem,
    unbounded: bool = False,
    max_evals: int | None = None,
    max_time: float | None = None,
    save_time_steps: bool = True,
    save_params_history: bool = True,
    save_batched_params_history: bool = False,
    save: list[str] | None = None,
    verbose: int = 0,
    print_every: int = 100,
    algorithm_str: str | None = None,
    save_to_file_every: int | None = None,
    display_mode: str = "live",
    unit_mapping: Callable | None = None,
    inverse_unit_mapping: Callable | None = None,
    hessian_batch_size: int = 1,
    checkpoint_format: str = "npz",
    checkpoint_dir: str | Path | None = None,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `problem` | `ContinuousProblem` | *required* | The optimization problem to wrap. |
| `unbounded` | `bool` | `False` | If `True`, the Objective maps unbounded parameters into problem bounds before evaluating `objective_function`, so algorithms can search in $(-\infty, +\infty)^\text{n\_params}$ space. If `False`, evaluates `objective_function` directly in bounded space. |
| `max_evals` | `int \| None` | `None` | Maximum number of function evaluations. `None` = unlimited. Batched evaluations are counted as how many parameters were given. |
| `max_time` | `float \| None` | `None` | Maximum wall-clock seconds beginning at the time `obj.start_logging()` was called. `None` = unlimited. |
| `save_time_steps` | `bool` | `True` | Record elapsed-time timestamp for each evaluation. |
| `save_params_history` | `bool` | `True` | Record the parameter vector at each evaluation. |
| `save_batched_params_history` | `bool` | `False` | Store full `(batch, n_params)` parameter arrays for batched evals instead of the reduced representative point. |
| `save` | `list[str] \| None` | `None` | List of advanced save tokens for recording additional / batched histories. Standard tokens: `"grad"`, `"hessian"`, `"eval_type"`, `"batched_loss"`, `"batched_grad"`, `"batched_hessian"`, `"batched"` (convenience alias expanding to the three batched tokens above). Aux diagnostics tokens: `"sensitivity_loss"`, `"penalty"`, `"is_feasible"`, `"power_values"`, `"violations"`, `"aux"` (convenience alias expanding to all five), plus per-field `batched_*` variants and `"batched_aux"`. On a problem exposing `objective_function_aux`, these tokens persist diagnostics from explicit `*_aux` methods and standard value-bearing methods. Derivative-only calls append aligned `None` entries. On a problem without `objective_function_aux`, aux tokens emit `RuntimeWarning` and automatic aux logging remains disabled; explicit aux methods still raise `RuntimeError`. The active configuration is recorded as a `SaveConfig` and restored from checkpoints so a resumed run keeps the original history schema. |
| `verbose` | `int` | `0` | Verbosity level. `0` = silent; `1` = periodic progress prints; `2` is WIP. |
| `print_every` | `int` | `100` | When `verbose ≥ 1`, print a progress summary every N evaluations. |
| `algorithm_str` | `str \| None` | `None` | If `None`, this is set by the algorithm via `prepare()` of `OptimizationAlgorithm`. Optional identifier string used in file names and logs. |
| `save_to_file_every` | `int \| None` | `None` | Automatically checkpoint every N evaluations. `None` disables auto-saving. The time spent saving is excluded from the elapsed-time clock. |
| `unit_mapping` | `Callable \| None` | `None` | Optional function mapping unbounded params to the **[0, 1] range**. Can be scalar (e.g. `jax.nn.sigmoid`) or element-wise vector. The Objective handles scaling to actual bounds: `bounded = lb + (ub - lb) * f(x)`. If omitted, the default sigmoid is used. |
| `inverse_unit_mapping` | `Callable \| None` | `None` | Inverse of the forward mapping, mapping [0, 1] -> unbounded space. The Objective normalises bounded params to [0, 1] before calling this: `unbounded = f_inv((bounded - lb) / (ub - lb))`. Must be provided whenever `unit_mapping` is provided. |
| `hessian_batch_size` | `int` | `1` | Number of Hessian columns to compute simultaneously via `vmap`. `1` (default) is the most memory-efficient (sequential `lax.map`); set to `n_params` for full `jax.hessian` parallelism. |
| `checkpoint_format` | `str` | `"npz"` | On-disk format for checkpoints. `"npz"` writes compressed NumPy archives; `"json"` writes a pickle-free, human-readable JSON file, useful when loading checkpoints from untrusted sources or when you want to inspect them by hand. No extra imports needed. |
| `checkpoint_dir` | `str \| Path \| None` | `None` | Root directory for checkpoint and output artifacts. Defaults to `./data/objective_run_data`. Pass a path to redirect all artifacts (e.g. to a scratch disk or a `tmp_path` in tests) without importing any storage class. |

To customise the storage stack beyond these two knobs (e.g. a custom serializer or a non-filesystem backend), subclass `Objective` and override `_build_storage`.

### Choosing `unbounded`

| `unbounded` | Objective function used | Example algorithms |
|-------------|-------------------------|--------------------|
| `False` | `problem.objective_function` | Random Search, PSO, CMA-ES, Bayesian Optimization |
| `True` | `Objective` mapping + `problem.objective_function` | Some gradient-based methods (Adam, L-BFGS, SA-GD, NA-Adam in their current implementations) |

### Choosing a bounded/unbounded mapping (important)

If you are new, use the defaults first:

- Leave both mapping arguments as `None`
- Set `unbounded=True` only if your optimizer expects unconstrained search space
- The default pair is sigmoid + inverse-sigmoid (logit)

Use a custom mapping only if you know exactly why you need it.

Rules:

- If you pass `unit_mapping`, you must also pass `inverse_unit_mapping`
- The forward mapping must produce values in **[0, 1]**; the Objective scales to actual bounds via `bounded = lb + (ub - lb) * f(x)`
- The inverse mapping receives values already normalised to [0, 1] by the Objective: `unbounded = f_inv((bounded - lb) / (ub - lb))`
- The inverse should satisfy approximately: `inverse(forward(x)) ≈ x` in the range you optimize over
- Both callables can be **scalar** functions (e.g. `jax.nn.sigmoid`) or **element-wise vector** functions: JAX broadcasts element-wise operations, so both work. The Objective uses `jax.vmap` for batching regardless

Minimal custom example (scalar function):

```python
import jax
from dfbench import Objective

# sigmoid maps (-inf, +inf) -> (0, 1): perfect for the [0,1] contract
obj = Objective(
    problem,
    unbounded=True,
    unit_mapping=jax.nn.sigmoid,
    inverse_unit_mapping=lambda x: jax.numpy.log(x / (1.0 - x)),
)
```

Element-wise vector example (different mapping per dimension):

```python
import jax.numpy as jnp

def forward(x):
    # Per-dimension [0,1] mapping; x is shape (n_params,)
    return jnp.where(x > 0, 1 - jnp.exp(-x), jnp.exp(x)) * 0.5 + 0.5

def inverse(x):
    x = jnp.clip(x, 1e-7, 1.0 - 1e-7)
    centered = 2.0 * (x - 0.5)
    return jnp.where(centered > 0, -jnp.log(1 - centered), jnp.log(centered + 1))

obj = Objective(
    problem,
    unbounded=True,
    unit_mapping=forward,
    inverse_unit_mapping=inverse,
)
```

You do **not** need to handle bounds scaling; the Objective does that automatically.

### Choosing what to save

The Objective always maintains an aligned `loss_history`: value-bearing calls store their loss, while derivative-only calls store a NaN placeholder because they do not return a primal loss. Three standard boolean flags control the most commonly toggled histories:

| Flag | Default | Effect |
|------|---------|--------|
| `save_time_steps` | `True` | Record one elapsed-time timestamp per admitted logged call |
| `save_params_history` | `True` | Record parameter vectors (reduced for batches) |
| `save_batched_params_history` | `False` | Store full `(batch, n_params)` parameter arrays instead of the reduced representative point |

For advanced combinations (gradients, Hessians, eval types, full batched arrays), pass a list of string tokens to `save`:

| Token | Effect |
|-------|--------|
| `"grad"` | Record gradients (one entry per logged call; batches reduced) |
| `"hessian"` | Record Hessians (one entry per logged call; batches reduced) |
| `"eval_type"` | Record one call-type bitmask per logged call |
| `"batched_loss"` | Store full `(batch,)` loss vectors instead of batch min |
| `"batched_grad"` | Store full `(batch, n_params)` gradient arrays |
| `"batched_hessian"` | Store full `(batch, n_params, n_params)` Hessian arrays |
| `"batched"` | Convenience alias expanding to the three `batched_*` tokens above |

Aux diagnostics tokens are recorded by the explicit `*_aux` evaluation methods and by the standard value-bearing methods (`value`, `value_and_grad`, `vmap_value`, `vmap_value_and_grad`, `value_grad_and_hessian`, `vmap_value_grad_and_hessian`). Auto-logging turns on when at least one aux token is in the `save` list and the problem exposes `objective_function_aux` (the built-in examples are `ConstrainedVoyagerProblem` and `UIFOProblem`). The standard methods then run the aux objective in the same forward pass and populate the aux histories without changing their return signatures (see [Auto-logging aux](#auto-logging-aux-from-standard-value-bearing-methods)). Each token controls one aux field, so enabling `is_feasible` does not force storing the bulky `power_values` arrays. On a problem without `objective_function_aux`, construction emits `RuntimeWarning`, standard methods continue through the scalar objective, and aux histories remain empty.

| Token | Effect |
|-------|--------|
| `"sensitivity_loss"` | Record the unpenalised sensitivity loss per logged call (reduced for batches; `None` when no aux is produced) |
| `"penalty"` | Record the summed penalty per logged call (reduced for batches; `None` when no aux is produced) |
| `"is_feasible"` | Record the physical feasibility flag per logged call (reduced for batches; `None` when no aux is produced) |
| `"power_values"` | Record per-group powers (hard, soft, detector) per logged call (reduced for batches; `None` when no aux is produced) |
| `"violations"` | Record per-constraint penalty values per logged call (reduced for batches; `None` when no aux is produced) |
| `"aux"` | Convenience alias expanding to the five non-batched aux tokens above |
| `"batched_sensitivity_loss"` | Store full batched sensitivity loss arrays |
| `"batched_penalty"` | Store full batched penalty arrays |
| `"batched_is_feasible"` | Store full batched feasibility bool arrays |
| `"batched_power_values"` | Store full batched per-group power arrays |
| `"batched_violations"` | Store full batched per-constraint violation arrays |
| `"batched_aux"` | Convenience alias expanding to the five `batched_*` aux tokens above |

When a `batched_*` aux token is off and the corresponding non-batched token is on, aux from every batched value-bearing call—including a singleton batch—is reduced to the representative point. Value-bearing calls select the index of the best loss, so the recorded `is_feasible` and `violations` reflect that best point. This matches the representative used for parameters, gradients, and Hessians.

```python
# Record gradients and full batched losses
obj = Objective(problem, save=["grad", "batched_loss"])

# Record everything (gradients, Hessians, eval types, all batched arrays)
obj = Objective(problem, save=["grad", "hessian", "eval_type", "batched"])

# Record aux diagnostics (feasibility, loss decomposition, powers) on a
# constrained problem, with full batched feasibility arrays
obj = Objective(problem, save=["aux", "batched_is_feasible"])
```

The active configuration is stored as a `SaveConfig` and embedded in every checkpoint's `RunMetadata`. On `load_run_data`, the Objective warns if the checkpoint config differs and adopts the stored config before restoring state, so resumed evaluations keep the run's original history schema.

### Storage (internal)

All file I/O (checkpointing, human-readable export) is handled internally by the modular `dfbench.core.storage` layer. The Objective assembles a `CheckpointManager` (with serializer, storage backend, and path resolver) and a `RunDataExporter` behind the scenes using sensible defaults; these components are **not** user-facing constructor parameters. The `save_to_file_every` argument is the only storage-related knob exposed to the user; it sets the periodic checkpoint cadence on the internal manager.

The storage components are still modular and individually testable (see [Storage & Checkpointing](Storage-and-Checkpointing)). Advanced users who need to swap a serializer, backend, or resolver can subclass `Objective` and override the internal assembly, or use the storage classes directly outside the Objective.

---

## Evaluation Methods

All evaluation methods automatically record their results in the internal history. They are the primary way algorithms should interact with the objective.

### Single-point evaluation

```python
obj.value(params)              # -> float
obj.grad(params)               # -> Array[n_params]
obj.hessian(params)            # -> Array[n_params, n_params]
obj.value_and_grad(params)     # -> (float, Array[n_params])
obj.value_grad_and_hessian(params)  # -> (float, Array[n_params], Array[n_params, n_params])
```

- `value(params)`: Evaluates the loss at `params`. Logs loss and params.
- `grad(params)`: Computes the gradient. The scalar primal is evaluated as part of differentiation, but its value is not returned or logged; `loss_history` receives a NaN placeholder.
- `hessian(params)`: Computes the exact Hessian. Logs Hessian and params, but **not** a loss value.
- `value_and_grad(params)`: Computes both in a single forward+backward pass. Logs all three. **Preferred when you need both loss and gradient** because it is more efficient than calling `value` and `grad` separately and it logs the loss.
- `value_grad_and_hessian(params)`: Computes loss, gradient, and Hessian together and logs all four.

### Batched evaluation

```python
obj.vmap_value(params_batch)              # -> Array[batch]
obj.vmap_grad(params_batch)               # -> Array[batch, n_params]
obj.vmap_hessian(params_batch)            # -> Array[batch, n_params, n_params]
obj.vmap_value_and_grad(params_batch)     # -> (Array[batch], Array[batch, n_params])
obj.vmap_value_grad_and_hessian(params_batch)
# -> (Array[batch], Array[batch, n_params], Array[batch, n_params, n_params])
```

Convenience aliases:

```python
obj.batched_value(...)            # same as vmap_value
obj.batched_grad(...)             # same as vmap_grad
obj.batched_hessian(...)          # same as vmap_hessian
obj.batched_value_and_grad(...)   # same as vmap_value_and_grad
obj.batched_value_grad_and_hessian(...)  # same as vmap_value_grad_and_hessian
```

Batched methods use `jax.vmap` and evaluate the entire batch as **one** history entry. The eval counter is incremented by the batch size. When `"batched_loss"` is not in the `save` list (default), only the batch minimum loss is stored.

### Aux evaluation

Available on problems that expose `objective_function_aux` (`ConstrainedVoyagerProblem` and `UIFOProblem` are the built-in examples). These methods return the loss plus an `aux` pytree dict carrying the loss decomposition, a physical `is_feasible` flag, per-constraint violations, and the raw per-group power arrays. See the [Power thresholds and aux diagnostics](#power-thresholds-and-aux-diagnostics) section for the aux schema.

```python
obj.value_aux(params)                     # -> (float, dict)
obj.value_and_grad_aux(params)            # -> (float, Array[n_params], dict)
obj.vmap_value_aux(params_batch)          # -> (Array[batch], dict)
obj.vmap_value_and_grad_aux(params_batch) # -> (Array[batch], Array[batch, n_params], dict)
```

- `value_aux(params)` returns `(loss, aux)`. The loss is logged into the standard loss history; aux fields are recorded into the per-field aux histories only when the matching save token is enabled.
- `value_and_grad_aux(params)` computes loss, gradient, and aux in one forward+backward pass via `jax.value_and_grad(..., has_aux=True)`. The gradient is taken with respect to the total returned loss, including any parameter-dependent penalty. `has_aux=True` carries the aux pytree through unchanged; derivatives of aux are neither returned nor logged.
- `vmap_value_aux` and `vmap_value_and_grad_aux` are the batched variants. Because `aux` is a JAX pytree, a batched call adds a leading batch dim to every leaf, including the `power_values` sub-arrays.
- The explicit aux methods raise `RuntimeError` when the wrapped problem does not expose `objective_function_aux` (for example `VoyagerProblem` and `VoyagerTuningProblem`). A custom `ContinuousProblem` may opt in by implementing the same aux objective contract.

Warmup helpers `warmup_value_aux`, `warmup_value_and_grad_aux`, `warmup_vmap_value_aux`, and `warmup_vmap_value_and_grad_aux` compile the aux callables before `start_logging()`. On problems without an aux objective they are a no-op (with a notice at `verbose >= 1`) rather than raising, so an algorithm that unconditionally warms up aux does not break on a non-constrained problem.

### Auto-logging aux from standard value-bearing methods

When one or more aux save tokens are enabled and the problem exposes `objective_function_aux`, the standard value-bearing methods (`value`, `value_and_grad`, `vmap_value`, `vmap_value_and_grad`, `value_grad_and_hessian`, `vmap_value_grad_and_hessian`) run the aux objective in the same forward pass and record the enabled aux diagnostics. No code change is needed in the optimization loop: a plain `obj.value(params)` call populates `is_feasible_history`, `sensitivity_loss_history`, and so on, in addition to `loss_history`.

This rebinds the internal value-bearing callables to the aux variants at bind time, so the aux pytree comes out of the same simulation that produced the loss. There is no second forward pass. The returned values keep their usual shapes (`value` still returns a scalar, `value_and_grad` still returns `(loss, grad)`); internally, exactly one aux pytree travels alongside the value result and is passed directly to `_log(..., aux=aux)`.

Grad-only and Hessian-only calls (`grad`, `hessian`, `vmap_grad`, `vmap_hessian`) differentiate the scalar objective but do not return or carry a primal loss/aux result. They therefore append `None` placeholders to enabled aux histories so those histories stay length-aligned with `loss_history`. The history properties preserve these `None` entries; `best_is_feasible` interprets one at the best-loss index as unavailable.

Auto-logging is active only when both conditions hold: at least one aux token is in the construction-time `save` list, and the problem exposes `objective_function_aux`. With no aux token, standard methods use the scalar primal and aux histories stay empty, so non-aux runs pay no aux-computation overhead. Supplying an aux token for a problem without an aux objective emits `RuntimeWarning` and falls back to the same scalar path; explicit aux methods still raise `RuntimeError`. Calling `set_penalty_fn` rebinds the cached callables so the selected value family stays in sync after a penalty swap.

```python
obj = Objective(problem, save=["is_feasible"])
obj.start_logging()
# Plain loop; is_feasible_history fills up alongside loss_history.
while not obj.budget_exceeded:
    loss, grad = obj.value_and_grad(params)
    params = params - 0.1 * grad
print(obj.best_is_feasible)
```

### Callable shorthand

```python
loss = obj(params)   # equivalent to obj.value(params)
```

### Unlogged raw value function

```python
obj.start_logging()
value_fn = obj.value_function()                 # follows obj.unbounded
value_fn = obj.value_function(unbounded=True)   # force unbounded mapping
value_fn = obj.value_function(unbounded=False)  # bounded problem objective
```

`value_function(unbounded=None)` returns a JAX-compatible scalar callable that does not itself log or increment the evaluation counter. It exists for optimizers that must call the value function inside their own JIT-compiled loop, such as Optax L-BFGS line search. The getter raises `RuntimeError` before `start_logging()`, so building and compiling a custom evaluation loop is inside the timed lifecycle.

When `unbounded=True`, the returned callable maps unbounded parameters into the problem bounds using the Objective's active mapping, then calls `problem.objective_function`. When `unbounded=False`, it calls `problem.objective_function` directly. Passing `None` uses the Objective's current `obj.unbounded` mode.

Because this callable is intentionally unlogged, pair it with `obj.log_evaluation(...)` after each completed optimizer step if the evaluation should count toward benchmark histories. For ordinary algorithm loops, prefer `obj.value(...)`, `obj.value_and_grad(...)`, or the batched evaluation methods.

### Unlogged raw value-and-aux function

```python
obj.start_logging()
aux_value_fn = obj.value_function_aux()  # follows obj.unbounded
if aux_value_fn is None:
    raise RuntimeError("this problem has no aux objective")

value_and_grad_aux_fn = jax.jit(
    jax.value_and_grad(aux_value_fn, has_aux=True)
)
_ = value_and_grad_aux_fn(params)  # custom JIT compilation is timed
(loss, aux), grad = value_and_grad_aux_fn(params)
obj.log_evaluation(params=params, loss=loss, grad=grad, aux=aux)
```

`value_function_aux(unbounded=None)` is the mapped, JAX-compatible counterpart to `value_function()`. After logging starts, it returns an unlogged callable whose invocation yields `(loss, aux)`, follows the same `unbounded` argument semantics, and returns `None` when the problem does not expose `objective_function_aux`. Before logging starts, the getter raises `RuntimeError`. This is the safe way for a custom JIT loop to obtain aux without bypassing the Objective's space mapping. `log_evaluation` persists only fields selected by aux save tokens.

### Manual logging

```python
obj.log_evaluation(params=..., loss=..., grad=..., hessian=..., aux=...)
```

For algorithms with custom JIT-compiled evaluation loops that use `obj.value_function(...)` or `obj.value_function_aux(...)` instead of calling `obj.value()` directly. Accepts `params`, `loss`, `grad`, `hessian`, and optional `aux` results and performs the same history recording. It raises `RuntimeError` before `start_logging()`. Supplied aux is persisted only for matching save tokens. In an aux-enabled run, omitting `aux` records `None` in each selected aux history. Manual logging never manufactures aux diagnostics.

---

## Lifecycle Methods

### `start_logging()`

Starts Objective logging and the wall-clock timer. **Call it after Objective-provided JIT warmup and before any evaluation or raw callable getter.** All `time_steps` and budget checks are relative to this moment. Calling it twice for the same run raises `RuntimeError`. Call `reset()` first to begin a new run.

```python
# Typical sequence
obj.warmup_value_and_grad()           # warmup
obj.start_logging()                    # timer starts NOW
while not obj.budget_exceeded:
    ...
```

### `warmup_*()`

`Objective` provides no-argument warmup helpers for every evaluation path:

```python
obj.warmup_value()
obj.warmup_grad()
obj.warmup_hessian()
obj.warmup_value_and_grad()
obj.warmup_value_grad_and_hessian()
obj.warmup_vmap_value(batch_size=10)
obj.warmup_vmap_grad(batch_size=10)
obj.warmup_vmap_hessian(batch_size=10)
obj.warmup_vmap_value_and_grad(batch_size=10)
obj.warmup_vmap_value_grad_and_hessian(batch_size=10)
```

Each helper executes the matching path **twice** on deterministic parameters and must be called before `start_logging()`. The batched variants accept a `batch_size` argument to match the batch size used during optimisation.

Before `start_logging()`, result-producing methods (`value*`, `grad`, `hessian`,
and `vmap_*`), raw callable getters (`value_function*`), and
`log_evaluation()` raise `RuntimeError`. Configuration and context remain
available: bounds, dimension, problem spec, optimization pairs, budget values,
random parameter generation, and pre-run setters do not require logging.

### `reset()`

Clears all histories, resets counters, and prepares for a completely fresh run. Does **not** change the problem, bounds, or budget limits.

### `set_seed(seed: int)`

Initializes the internal JAX PRNG key. Subsequent calls to `random_params()`, `random_params_bounded()`, and `random_params_unbounded()` consume and split this key automatically, guaranteeing identical initial samples across runs with the same seed.

This is to facilitate uniform initialization across algorithms.

```python
obj.set_seed(42)
p1 = obj.random_params_bounded(100)   # deterministic
p2 = obj.random_params_bounded(100)   # different from p1 but reproducible
obj.set_seed(42)
p3 = obj.random_params_bounded(100)   # identical to p1
```

### `set_space_mode(unbounded, unit_mapping=None, inverse_unit_mapping=None)`

Switches between bounded and unbounded mode before optimization starts.

- Must be called before `start_logging()`
- Re-binds all internal JAX evaluation paths (`value`, `grad`, `hessian`, all `vmap_*`)
- Can optionally replace the mapping pair at the same time
- Custom mappings follow the same [0, 1] contract as the constructor: the forward function maps to [0, 1], the Objective handles bounds scaling

```python
# default sigmoid mapping
obj.set_space_mode(True)

# custom mapping pair (scalar functions work)
import jax
obj.set_space_mode(
    True,
    unit_mapping=jax.nn.sigmoid,
    inverse_unit_mapping=lambda x: jax.numpy.log(x / (1.0 - x)),
)
```

### `set_penalty_fn(fn)`

Sets the wrapped problem's penalty function and re-binds all internal JAX evaluation paths.

For problems that opt into the power-penalty contract (`ConstrainedVoyagerProblem`, `UIFOProblem`), this forwards to `problem.set_penalty_fn(fn)`, which updates the problem's penalty callable and rebuilds both its JIT-compiled scalar and aux objectives. The Objective then re-binds its own cached `value` / `grad` / `hessian` / `vmap_*` callables, mirroring the tail of `set_space_mode`.

- Must be called before `start_logging()`
- Raises `RuntimeError` if the wrapped problem does not opt into the power-penalty contract. The opt-in marker is the class attribute `_supports_power_penalty`; problems without a power-constraint path (`VoyagerProblem`, `VoyagerTuningProblem`, and any non-optical `ContinuousProblem`) leave it `False` and reject the call rather than silently rebuilding, even though they inherit the method from `OpticalSetupProblem`
- Composes with `set_space_mode` in either order before `start_logging()`

```python
from dfbench.problems import relu_penalty, zero_penalty

obj.set_penalty_fn(relu_penalty)   # use the raw ReLU penalty
obj.set_penalty_fn(zero_penalty)   # disable the penalty term entirely
```

The current penalty function is exposed via the read-only `obj.penalty_fn` property (`None` for problems without penalty support).

### Power thresholds and aux diagnostics

`obj.power_thresholds` returns a dict with the constant per-group thresholds (`"hard"`, `"soft"`, `"detector"`) for problems that opt into the penalty contract, or `None` otherwise. Thresholds are physical constants; they do not change across evaluations or after `set_penalty_fn`.

The constrained problems also expose a JIT-compiled `objective_function_aux(params)` alongside `objective_function`. It returns `(loss, aux)` where `aux` is a pytree dict carrying the loss decomposition and physical diagnostics:

| Key | Shape | Description |
|-----|-------|-------------|
| `sensitivity_loss` | scalar | The unpenalised sensitivity loss. |
| `penalty` | scalar | The summed penalty contribution. |
| `is_feasible` | scalar bool | `True` iff every per-group power is at or below its threshold. This is a physical check, independent of the active `power_penalty_fn` preset. |
| `violations` | `(n_constraints,)` | Per-constraint penalty values. |
| `power_values` | dict with `hard`, `soft`, `detector` leaves | Raw per-group power arrays. |

`aux` is a JAX pytree, so `objective_function_aux` vmapps cleanly: a batched call adds a leading batch dim to every leaf, including the `power_values` sub-arrays. The Objective-level explicit `*_aux` wrappers and standard value-bearing methods thread aux through logging and the save-token system; see the [Aux evaluation](#aux-evaluation) subsection and the aux tokens in [Choosing what to save](#choosing-what-to-save).

---

## Random Sampling

### `random_params(n_samples=1, rng_key=None)`

Returns random samples from the active Objective space: bounded when `obj.unbounded` is `False`, unbounded when `obj.unbounded` is `True`. Prefer this in algorithms after calling `prepare()` when sampling should follow the configured space mode.

### `random_params_bounded(n_samples=1, rng_key=None)`

Returns uniform random samples inside `problem.bounds`.

| Argument | Default | Description |
|----------|---------|-------------|
| `n_samples` | `1` | How many vectors to draw. Returns shape `(n_params,)` when 1, `(n_samples, n_params)` otherwise. |
| `rng_key` | `None` | Optional manual JAX key. If `None`, uses the internal key set by `set_seed()`. |

### `random_params_unbounded(n_samples=1, rng_key=None)`

Generates samples uniform in the bounded space then maps them to unbounded space using:

- your custom `inverse_unit_mapping` if provided: the Objective normalises bounded samples to [0, 1] first, then calls your inverse
- otherwise the default inverse sigmoid (logit)

With the matching forward mapping, this round-trip holds:

```python
bounded ≈ lb + (ub - lb) * forward(random_params_unbounded(...))
```

---

## Properties

### Problem & Configuration

| Property | Type | Description |
|----------|------|-------------|
| `bounds` | `Array[2, n_params]` | Lower and upper parameter bounds (or $\pm\infty$ when unbounded). |
| `n_params` | `int` | Number of optimizable parameters. |
| `optimization_pairs` | `list` | Defensive-copy component/property context aligned by parameter index: `optimization_pairs[i]` describes parameter `i`. Usually a `(component_name, property_name)` tuple. Coupled parameters may contain a list of tuples. |
| `problem_name` | `str` | Display name of the wrapped problem. |
| `problem_spec` | `dict[str, Any]` | Fresh JSON-safe typed spec (`type`, `version`, `params`) sufficient to reconstruct the wrapped problem. Contains problem configuration, not run results. |
| `penalty_fn` | `Callable \| None` | The wrapped problem's penalty function, or `None` if the problem does not expose one. Update it via `set_penalty_fn(fn)`. |
| `power_thresholds` | `dict[str, float] \| None` | Per-group power thresholds (`hard`, `soft`, `detector`) for problems that opt into the penalty contract, or `None`. Constants; do not change across evaluations. |

### Budget Tracking

| Property | Type | Description |
|----------|------|-------------|
| `eval_count` | `int` | Total evaluations so far. |
| `max_evals` | `int \| None` | The evaluation budget, or `None` if unlimited. |
| `max_time` | `float \| None` | The wall-clock time budget in seconds, or `None` if unlimited. |
| `evals_left` | `int \| None` | Remaining evaluation budget. `None` if unlimited. |
| `evals_exceeded` | `bool` | Whether the evaluation cap has been reached. |
| `evals_progress_fraction` | `float` | Fraction of eval budget consumed (0-1). |
| `time_elapsed` | `float` | Seconds since `start_logging()`. |
| `time_left` | `float \| None` | Seconds remaining. `None` if unlimited. |
| `time_exceeded` | `bool` | Whether the time cap has been reached. |
| `time_progress_fraction` | `float` | Fraction of time budget consumed (0-1). |
| `budget_left_fraction` | `float` | Fraction of the tightest budget remaining. `min(1 - time_progress, 1 - evals_progress)`, considering only budgets that are set. 1.0 when no budget is configured. |
| `budget_progress_fraction` | `float` | Fraction of the tightest budget consumed (`1 - budget_left_fraction`). 0.0 when no budget is configured. |
| `budget_exceeded` | `bool` | `True` when **any** budget (time **or** evals) is exhausted. This is the main loop-termination check. |
| `save_every` | `int \| None` | Periodic checkpoint cadence in evaluations, or `None` if disabled. |

### Best Results

| Property | Type | Description |
|----------|------|-------------|
| `best_loss` | `float \| None` | Lowest loss observed. `None` before the first evaluation. |
| `best_params` | `Array \| None` | Raw parameters at `best_loss` (may be in unbounded space). |
| `best_params_bounded` | `Array \| None` | Best parameters mapped to bounded space via the active mapping (custom mapping if configured, otherwise sigmoid). **Use this for final output.** |
| `best_eval_index` | `int \| None` | Index into the loss history holding the best loss, or `None` before the first improvement. |
| `best_batch_index` | `int \| None` | Within-batch index of the best loss when it came from a batched evaluation, or `None` for single-point evals. |
| `best_is_feasible` | `bool \| None` | Feasibility of the best-loss point, or `None` when neither `is_feasible` nor `batched_is_feasible` was enabled, no evaluation has improved yet, or a manually logged best point omitted aux. Uses the physical `power <= threshold` check, so it stays meaningful even when the penalty is disabled with `zero_penalty`. For batched best losses, the feasibility of the winning batch element is returned when `batched_is_feasible` storage is on; otherwise the per-call reduced entry is used. |

### Current State

| Property | Type | Description |
|----------|------|-------------|
| `current_loss` | `float \| Array \| None` | Loss from the most recent evaluation. |
| `current_params` | `Array \| None` | Parameters from the most recent evaluation. |

### Raw History

These properties return **copies** to prevent external mutation.

| Property | Type | Description |
|----------|------|-------------|
| `loss_history` | `list` | All recorded losses (may contain batched arrays). |
| `grad_history` | `list` | All recorded gradients (if saving was enabled). |
| `hessian_history` | `list` | All recorded Hessians (if saving was enabled). |
| `params_history` | `list` | All recorded parameter vectors (raw space, i.e. as it was given to the `Objective`). |
| `params_history_bounded` | `list` | Params history mapped to bounded space. |
| `time_steps` | `list[float]` | Elapsed time at each recorded evaluation. |
| `sensitivity_loss_history` | `list` | Per-logged-call unpenalised sensitivity loss when selected; `None` for calls without aux. |
| `penalty_history` | `list` | Per-logged-call summed penalty when selected; `None` for calls without aux. |
| `is_feasible_history` | `list` | Per-logged-call physical feasibility flag when selected; `None` for calls without aux. |
| `violations_history` | `list` | Per-logged-call constraint violation arrays when selected; `None` for calls without aux. |
| `power_hard_history` | `list` | Per-logged-call hard-group power arrays when `power_values` storage is selected; `None` for calls without aux. |
| `power_soft_history` | `list` | Per-logged-call soft-group power arrays when `power_values` storage is selected; `None` for calls without aux. |
| `power_detector_history` | `list` | Per-logged-call detector-group power arrays when `power_values` storage is selected; `None` for calls without aux. |

Each enabled aux history is index-aligned with `loss_history`: value-bearing standard and explicit aux calls append the selected diagnostics, while derivative-only calls and `log_evaluation(..., aux=None)` append `None`. Aux histories whose save token is disabled remain empty.

### Reduced History

**Rationale:** Batched evaluations produce `(batch, ...)` shaped entries. Downstream analysis (benchmarking, plotting) expects flat lists of scalars/vectors. The `*_reduced` properties collapse each batch to a single representative value:

1. Select the entry (for loss, grad, Hessian and param) with the minimum loss if available for that step.
2. Else select the entry with the minimum gradient norm.
3. Else select the entry with the minimum Hessian norm.
4. Else take the first element.

| Property | Type | Description |
|----------|------|-------------|
| `loss_history_reduced` | `list[float]` | Losses with batches reduced to `nanmin`. |
| `params_history_reduced` | `list[Array \| None]` | Params with batches reduced per the rule above. |
| `params_history_reduced_bounded` | `list[Array \| None]` | Reduced params in bounded space. |
| `grad_history_reduced` | `list[Array \| None]` | Grads with batches reduced. |
| `hessian_history_reduced` | `list[Array \| None]` | Hessians with batches reduced. |

### Progress Counters

| Property | Type | Description |
|----------|------|-------------|
| `improvement_count` | `int` | How many times `best_loss` was improved. |
| `evals_since_improvement` | `int` | Evaluations since the last improvement, useful for patience-based early stopping. |

---

## I/O Methods

All file I/O is handled internally by the modular `dfbench.core.storage` layer (see [Storage & Checkpointing](Storage-and-Checkpointing)). The Objective builds and applies the canonical `RunState` data contract; the serializer, backend, and resolver are assembled internally with sensible defaults.

### `save_run_data(algorithm_name=None, filepath=None, hyper_param_str=None) -> Path`

Saves the full optimization state to a checkpoint file via the internal `CheckpointManager.save()`. The serializer (default `NpzCheckpointSerializer`) encodes a `RunState` snapshot; the backend (default `LocalFilesystemBackend`) writes it **atomically** (temp file in the same directory + `os.replace`), so an interrupted job never leaves a half-written file. If `algorithm_name` is not provided it defaults to `self.algorithm_str` (or `"unknown"`).

The checkpoint embeds `RunMetadata` (problem/algo/budget identity, `SaveConfig`, and the problem's typed `ProblemSpec` container; see [Problems](Problems)), so the file is fully self-describing.

Default path (built by `RunPathResolver`, then anchored to a directory by `LocalFilesystemBackend`): `<checkpoint_dir>/{budget_dir}/{problem}_{algo}_{hyper_param_str}_{timestamp}.npz`, absolute on disk. `checkpoint_dir` defaults to `./data/objective_run_data`. Set `RunPathResolver(algo_directory=True)` to add an `{algo}_{hyper_param_str}` subdirectory under `{budget_dir}`.

The first save without explicit overrides caches the path; subsequent periodic saves overwrite the same file.

### `load_run_data(filepath)`

Restores all tracking state from a checkpoint via `CheckpointManager.load()` -> `Objective._apply_run_state()`. Adjusts `start_time` so that `time_elapsed` continues seamlessly from where the checkpoint left off. The loaded path is cached so a later `save_run_data()` overwrites the same file. If the checkpoint's `SaveConfig` differs from the constructor configuration, `RuntimeWarning` is emitted and the checkpoint configuration is adopted before state is applied. When the problem supports aux logging, empty legacy aux histories selected by that config are backfilled with one `None` entry per prior logged call before evaluation resumes.

The originating `Problem` can be rebuilt from the embedded `problem_spec`:

```python
from dfbench.core.storage import CheckpointManager
from dfbench.core.problem import ProblemSpec, build_problem_from_spec

state = obj._checkpoint_manager.load(path)
spec = CheckpointManager.extract_problem_spec(state)  # -> dict | None
if spec is not None:
    problem = build_problem_from_spec(ProblemSpec.from_dict(spec))
```

### `output_to_files(hyper_param_str="", hyper_param_str_in_filename=True, *, write_parameters_json=True, write_losses_json=True, write_losses_png=True, write_sensitivity_png=True) -> Path`

Writes human-readable outputs via `RunDataExporter.export()`, which derives everything from a `RunState` snapshot (not a second write path):
- JSON with best parameters (bounded space)
- JSON with loss history
- PNG plot of the loss curve
- (For optical problems) PNG plot of the sensitivity curve vs. target

Output directory (built by the exporter): `data/problem_output/{problem_name}/{algorithm_str}_{hyper_param_str}/` (collapsing to just `{algorithm_str}` when there is no hyperparameter string). When `hyper_param_str` is not passed explicitly, it is read from the run state's `metadata.hyper_param_str`.

Each artifact is independently optional; pass the corresponding `write_*` flag as `False` to skip that file. The output directory is still created and returned regardless of which artifacts are written. The `write_sensitivity_png` flag is an additional gate on top of the existing optical-problem condition; passing `True` does not force a sensitivity plot on a non-optical problem.

### `get_summary() -> dict`

Returns a snapshot dictionary:

```python
{
    "eval_count": int,
    "time_elapsed": float,
    "best_loss": float | None,
    "current_loss": float | None,
    "improvement_count": int,
    "evals_since_improvement": int,
    "budget_exceeded": bool,
    "time_exceeded": bool,
    "evals_exceeded": bool,
}
```

---

## Internal Logging Details

Every evaluation method follows the same pipeline internally:

1. **Execute** the JAX function (`_func`, `_value_and_grad_func`, `_vmap_func`, etc.)
2. **`_log(params, loss, grad, hessian, aux)`**: takes one elapsed-time snapshot, asks `_log_evals()` to admit and atomically record the call, appends the timestamp only when admitted, then invokes `_log_to_file()`.
3. **`_log_evals(params, loss, grad, hessian, aux, time_exceeded)`**: checks the evaluation/time budget, resolves selected aux leaves before mutating histories, records all aligned histories, updates best/stagnation state, and returns whether the call was admitted.
4. **`_log_to_file()`**: calls `CheckpointManager.tick(eval_count, state_factory)`, which checks the cadence (`save_every`, set from `save_to_file_every`), lazily builds a `RunState` only when a checkpoint is due, saves it through the internal `StorageBackend`, and returns the wall-clock duration of the save. The Objective advances `_start_time` by that duration so the checkpoint write does not consume wall-clock budget.

> **Important:** These are private methods; do not call `_log()`, `_log_evals()`, or `_log_to_file()` directly from algorithm code. If you want manual logging, use the public `log_evaluation(params, loss, grad=None, hessian=None, aux=None)` method instead, which delegates to `_log()`. See the [JIT-compiled loop guide](Implementing-a-New-Algorithm#custom-jit-compiled-loops-with-log_evaluation) for details.

Budget enforcement happens *after* the evaluation returns. This means the algorithm always receives a valid result, but once any budget is exceeded the history stops growing and `budget_exceeded` becomes `True`.

When a batch evaluation (`vmap_*`) would push `eval_count` past `max_evals`, the evaluations are counted but *not logged*, preserving history alignment and setting the `budget_exceeded` flag to `True`. Because admission happens before history mutation, no loss, parameter, derivative, aux, or time entry needs to be rolled back. Reducing population as `evals_left` nears zero avoids evaluating an unrecorded batch.
