# Implementing a New Algorithm

This is the primary guide for contributors adding a new optimization algorithm to dfbench. The framework is designed to make this as painless as possible: you write the optimization logic, `Objective` handles everything else.

> **Note:** This page is only for **dfbench contributors**. For the Learn2Design-2026 competition, you do **not** need to contribute your own algorithm to the main dfbench repository. Competition participants should follow [docs/submission.md](../submission.md) instead.

---

## The Contract

Every algorithm must:

1. **Subclass** `OptimizationAlgorithm`
2. **Declare** `algorithm_str` and `algorithm_type`
3. **Implement** `optimize(objective, ...) -> None`
4. **Use `Objective`** for all function evaluations
    - *Unless* you use your own JIT-compiled loop, then manually log calls via `Objective.log_evaluation(...)` afterwards. This adds negligible overhead relative to the objective function.

That's it. No manual timing, history management or file I/O necessary.

---

## Step 1: Create the File

Place your algorithm in the appropriate subdirectory:

```
src/dfbench/algorithms/
├── derivative_free/     <- direct search and non-gradient local solvers
├── global_search/       <- stochastic global optimizers
├── evolutionary/        <- population-based methods (PSO, ES)
├── gradient_based/
│   ├── optax/           <- Optax optimizer wrappers (subclass OptaxAlgorithm)
│   ├── scipy/           <- SciPy minimize wrappers (subclass ScipyMinimizeAlgorithm)
│   ├── custom_jax.py    <- native-JAX custom/hybrid gradient methods
│   └── *.py             <- custom optimization loops
├── surrogate_based/     <- builds a surrogate model (BO, kNN, etc.)
└── generative/          <- generative models (VAE, diffusion, etc.)
```

For example: `src/dfbench/algorithms/evolutionary/my_algorithm.py`

---

## Step 2: Implement the Algorithm

Here is a complete, minimal template:

```python
import secrets
import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from dfbench.core.algorithm import OptimizationAlgorithm, AlgorithmType
from dfbench.core.objective import Objective


class MyAlgorithm(OptimizationAlgorithm):
    """One-line description of your algorithm.

    Longer description, references, etc.
    """

    algorithm_str: str = "my_algorithm"          # unique identifier
    algorithm_type: AlgorithmType = AlgorithmType.GLOBAL_SEARCH  # match the algorithms/ subfolder

    def __init__(self, batch_size: int = 50) -> None:
        """Initialize with algorithm-level meta-parameters.

        These are parameters intrinsic to the algorithm that don't change
        between runs (e.g. batch size, network architecture).
        """
        self.batch_size = batch_size

    def optimize(
        self,
        objective: Objective,
        max_iterations: int | None = None,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        # ---- your hyperparameters below ----
        my_hyperparam: float = 1.0,
        patience: int = 1000,
        **kwargs,
    ) -> None:
        """Run the optimization.

        Args:
            objective: Pre-configured Objective instance.
            max_iterations: Max algorithm iterations (not evals). None = budget only.
                For algorithms where each iteration performs exactly one evaluation
                (e.g. gradient-based methods), omit this parameter -
                `obj.budget_exceeded` already handles it.
            init_params: Optional starting point.
            random_seed: Seed for reproducibility.
            my_hyperparam: Description of your hyperparameter.
            patience: Stop after N iterations without improvement.
        """
        # ─── 1. Setup references ───
        obj = objective

        # Sets unbounded mode, algorithm_str, seeds np/JAX, returns resolved seed + JAX key
        random_seed, key = self.prepare(
            obj,
            unbounded=False,
            random_seed=random_seed,
        )
        # torch.manual_seed(random_seed)  # add if you use PyTorch

        # ─── 3. Initialize parameters ───
        if init_params is None:
            params = obj.random_params_bounded(n_samples=self.batch_size)
        else:
            params = init_params

        # ─── 4. JIT warmup ───
        # This compiles the JAX computation graph. Do it BEFORE start_logging()
        # so compilation time doesn't count against the 4 hour budget.
        obj.warmup_vmap_value(batch_size=self.batch_size)

        # ─── 5. Start logging (starts the clock) ───
        obj.start_logging()

        # ─── 6. Optimization loop ───
        iteration = 0
        while not obj.budget_exceeded:
            if max_iterations is not None and iteration >= max_iterations:
                break

            # Evaluate
            losses = obj.vmap_value(params)    # automatically logged!

            # Your update logic
            key, subkey = jax.random.split(key)
            # ... update params using losses, subkey, my_hyperparam, etc. ...

            # Early stopping (optional)
            if obj.evals_since_improvement > patience:
                break

            iteration += 1

        # ─── 7. Done ───
        # The Objective is mutated in place. The caller accesses results
        # from the same instance it passed in. No return needed.
```

---

## Step 3: Register the Algorithm

Add your import to two files:

**`src/dfbench/algorithms/<category>/__init__.py`:**

```python
from dfbench.algorithms.evolutionary.my_algorithm import MyAlgorithm
```

**`src/dfbench/algorithms/__init__.py`:**

```python
from dfbench.algorithms.evolutionary.my_algorithm import MyAlgorithm

__all__ = [
    ...,
    "MyAlgorithm",
]
```

---

## The `optimize()` Blueprint in Detail

The base class `OptimizationAlgorithm.optimize()` contains a commented blueprint showing every step. Here's what each step does and why:

### 1. Setup references

```python
obj = objective
random_seed, key = self.prepare(
    obj,
    unbounded=False,
    random_seed=random_seed,
)
```

The problem instance wrapped by the `Objective` is private. Use the public
Objective API directly: `obj.bounds` for the `(2, n_params)` bounds,
`obj.n_params` for the dimension, `obj.problem_name` for the display name,
`obj.problem_spec` for the JSON-safe reconstructive problem configuration,
`obj.optimization_pairs[i]` for the component/property context of parameter
`i`, and
`obj.penalty_fn` / `obj.power_thresholds` for the penalty contract.

`prepare()` is called as `prepare(obj, unbounded, random_seed, algorithm_str=None, **kwargs)`. It sets `obj.unbounded`, `obj.algorithm_str`, seeds `np.random` and JAX, and returns `(random_seed, key)`. If `random_seed=None` is passed, a seed is generated via system entropy. You can also configure the Objective manually instead of calling `prepare()`.

**Choose `unbounded`:**
- `False` if your algorithm naturally handles bound constraints (evolutionary, derivative-free, global-search, surrogate-based)
- `True` if you want smooth unconstrained space where gradients never hit box boundaries (via sigmoid transform)

### 2. Set random seeds

`prepare()` handles `np.random`, JAX, and the seed print. If your algorithm uses PyTorch or another framework, seed it with the returned value:

```python
random_seed, key = self.prepare(
    obj,
    unbounded=False,
    random_seed=random_seed,
)
torch.manual_seed(random_seed)   # only needed for PyTorch-based libraries
```

### 3. Initialize parameters

For **bounded** space:
```python
params = obj.random_params_bounded()              # shape: (n_params,)
batch = obj.random_params_bounded(n_samples=100)  # shape: (100, n_params)
```

For **unbounded** space:
```python
params = obj.random_params_unbounded()             # shape: (n_params,)
```

### 4. JIT warmup

```python
obj.warmup_value()                       # single eval path
obj.warmup_value_and_grad()              # when using gradients
obj.warmup_vmap_value(batch_size=100)    # for batched methods (match your batch size)
```

Warmup can take seconds because it triggers JAX compilation. Do this **before** `start_logging()` so the compilation time is not counted against the 4-hour budget. The `warmup_*()` helpers use deterministic params internally and run the corresponding path twice. Single-point helpers take no arguments; batched helpers take the batch size used by your algorithm.

### 5. Start logging

```python
obj.start_logging()
```

This starts Objective logging and the 4-hour wall-clock timer. Objective-provided
`warmup_*()` calls before this line are free. Evaluation methods, raw callable
getters, and `log_evaluation()` raise `RuntimeError` until this line has run.

### 6. Main loop

```python
while not obj.budget_exceeded:
    loss = obj.value(params)           # or obj.value_and_grad, obj.vmap_value, etc.
    # ... your algorithm logic ...
```

`budget_exceeded` returns `True` when either the time or evaluation budget is exhausted. Once exceeded, further evaluations still work (JAX functions still run) but their results are **not logged**.

### 7. Done

`optimize()` returns `None`. The `Objective` was mutated in place; the caller (benchmark harness or user script) accesses results from the same instance it passed in. This follows the Python convention that in-place mutating methods return `None` (like `list.sort()`).

---

## Choosing the Right Evaluation Method

| Method | When to use | Logs |
|--------|-------------|------|
| `obj.value(params)` | Need loss only, single point | loss, params |
| `obj.grad(params)` | Need gradient only (rare) | grad, params (no loss!) |
| `obj.hessian(params)` | Need exact second-order information | hessian, params (no loss!) |
| `obj.value_and_grad(params)` | Gradient-based optimization | loss, grad, params |
| `obj.value_grad_and_hessian(params)` | Newton-style / second-order optimization | loss, grad, hessian, params |
| `obj.vmap_value(batch)` | Population evaluation | batch losses, batch params |
| `obj.vmap_value_and_grad(batch)` | Batched gradient-based | batch losses, grads, params |
| `obj.vmap_hessian(batch)` | Batched second-order optimization | batch hessians, batch params |
| `obj.vmap_value_grad_and_hessian(batch)` | Batched second-order optimization | batch losses, grads, hessians, params |
| `obj.value_function(...)` | Raw JAX callable for custom JIT loops | nothing; use `log_evaluation` afterwards |
| `obj.value_function_aux(...)` | Raw `(loss, aux)` JAX callable for custom JIT loops | nothing; use `log_evaluation` afterwards |
| `obj.log_evaluation(...)` | Custom JIT'd loop | supplied values; aux only for matching save tokens |

**Important:** `obj.grad()` and `obj.hessian()` do **not** log a loss value or produce aux diagnostics. If aux histories are enabled, derivative-only methods append aligned `None` entries. With an aux save token, `obj.value_and_grad()` and `obj.value_grad_and_hessian()` auto-log primal aux while retaining their standard return signatures. Use `obj.value_and_grad_aux()` when the algorithm itself must receive aux.

### Custom JIT-compiled loops with `log_evaluation()`

Some optimizers (e.g. Optax's L-BFGS) need to call `value_and_grad` *inside* a JIT-compiled function, for instance because the optimizer's line-search requires the raw value function. In that case you can't use `obj.value_and_grad()` (which has Python-side logging). Instead:

1. Get an unlogged scalar callable from `obj.value_function(...)`, or an aux-aware callable from `obj.value_function_aux(...)` whose invocation yields `(loss, aux)`
2. Build your own JIT-compiled step
3. After each step, call `obj.log_evaluation(params, loss, grad, hessian=None, aux=None)` to record the results

Call `obj.start_logging()` before step 1. Unlike Objective-provided
`warmup_*()` paths, construction and compilation of a custom raw-callable path
are timed because the raw getter is intentionally unavailable before logging.

`obj.value_function(unbounded=None)` follows the Objective's active space mode by default. Pass `unbounded=True` when the JIT loop works in unbounded coordinates and needs Objective's mapping into problem bounds; pass `unbounded=False` for bounded coordinates. `obj.value_function_aux(...)` follows the same mapping rules and returns an unlogged callable whose invocation yields `(loss, aux)`, or `None` when the problem has no aux objective. Neither callable logs anything.

`log_evaluation()` never computes missing aux diagnostics. If the Objective has aux save tokens and the custom loop does not pass `aux`, each selected aux history receives `None` for that call. A custom loop that already computed a conforming aux pytree may pass it explicitly.

To compute mapped aux inside the custom loop, preserve it as auxiliary output during differentiation:

```python
obj.start_logging()
value_aux_fn = obj.value_function_aux()
if value_aux_fn is None:
    raise RuntimeError("this problem has no aux objective")
value_and_grad_aux_fn = jax.jit(jax.value_and_grad(value_aux_fn, has_aux=True))

_ = value_and_grad_aux_fn(params)  # custom JIT compilation is timed
(loss, aux), grads = value_and_grad_aux_fn(params)
obj.log_evaluation(params, loss, grads, aux=aux)
```

Only aux fields selected in the Objective's `save` configuration are persisted.

```python
# Start timing before obtaining the raw function.
obj.start_logging()
value_fn = obj.value_function(unbounded=True)
value_and_grad_fn = jax.value_and_grad(value_fn)

@jax.jit
def _step(params, opt_state):
    loss, grads = value_and_grad_fn(params)
    updates, new_state = optimizer.update(grads, opt_state, params, ...)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_state, loss, grads

_ = _step(params, state)   # custom JIT compilation is timed

while not obj.budget_exceeded:
    prior_params = params
    params, state, loss, grads = _step(params, state)
    obj.log_evaluation(prior_params, loss, grads)  # public API for manual logging
```

> **Do NOT call** `obj._log()`, `obj._log_evals()`, or `obj._log_to_file()` directly; these are private methods. `log_evaluation()` delegates to `_log()` which coordinates all internal logging.

See `LBFGSGD` in `src/dfbench/algorithms/gradient_based/lbfgs_gd.py` for a complete working example.

---

## Bounded vs. Unbounded: Detailed Guide

### Bounded optimization (`unbounded=False`)

- Parameters live in `[lower, upper]` for each dimension.
- Use `obj.value()` or `obj.vmap_value()`.
- Algorithms must keep parameters within bounds (via clamping, constrained sampling, etc.).
- Useful for evolutionary, derivative-free, global-search, and surrogate-based methods.

Be careful!
- Ask for a `batch_size` in the `__init__()` and try different sizes.
- Changing the `batch_size` causes a recompile which can take a while.
```python
random_seed, key = self.prepare(
    obj,
    unbounded=False,
    random_seed=random_seed,
)
params = obj.random_params_bounded(n_samples=100)
losses = obj.vmap_value(params)
```

### Unbounded optimization (`unbounded=True`)

- Parameters live in $(-\infty, +\infty)$.
- The objective internally applies sigmoid bounding: $\text{bounded} = \text{lb} + (\text{ub} - \text{lb}) \cdot \sigma(\text{unbounded})$.
- To get final results in the physical space: `obj.best_params_bounded`.

```python
random_seed, key = self.prepare(
    obj,
    unbounded=True,
    random_seed=random_seed,
)
params = obj.random_params_unbounded()
loss, grad = obj.value_and_grad(params)
```

If you need a different transform than sigmoid, configure it explicitly before logging starts. Custom mappings must map to the [0, 1] range; the Objective handles scaling to actual bounds (`bounded = lb + (ub - lb) * f(x)`):

```python
import jax

# Scalar function: works on floats and arrays alike
obj.set_space_mode(
    True,
    unit_mapping=jax.nn.sigmoid,
    inverse_unit_mapping=lambda x: jnp.log(x / (1.0 - x)),
)
```

Important:

- Always pass both functions together (forward and inverse)
- The forward maps to [0, 1]; the inverse maps [0, 1] -> unbounded. Bounds scaling is handled by `Objective`
- Functions can be scalar (e.g. `jax.nn.sigmoid`) or element-wise vector; both work because JAX broadcasts element-wise operations. Batching is handled by `Objective` via `jax.vmap`
- `obj.best_params_bounded` and `obj.random_params_unbounded()` then follow your custom mapping pair

---

## Using External Libraries

Many algorithms wrap external optimization libraries (EvoX, BoTorch, Optax). Here's the pattern:

### PyTorch-based libraries (EvoX, BoTorch)

```python
from dfbench import t2j, j2t

# Convert JAX -> PyTorch for the library
params_torch = j2t(params_jax)

# Convert PyTorch -> JAX for evaluation
params_jax = t2j(params_torch)
losses_jax = obj.vmap_value(params_jax)
losses_torch = j2t(losses_jax)
```

The conversion goes through NumPy and adds negligible overhead relative to evaluation time.

### JAX-based libraries (Optax)

No conversion needed; Optax operates directly on JAX arrays:

```python
import optax
optimizer = optax.adam(learning_rate=0.1)
state = optimizer.init(params)

loss, grad = obj.value_and_grad(params)
updates, state = optimizer.update(grad, state, params)
params = optax.apply_updates(params, updates)
```

---

## Checklist Before Submitting

- [ ] Algorithm subclasses `OptimizationAlgorithm`
- [ ] `algorithm_str` is set to a unique identifier
- [ ] `algorithm_type` is set correctly
- [ ] `optimize()` accepts `objective: Objective` as first arg
- [ ] `optimize()` returns `None` (the `Objective` is mutated in place)
- [ ] All evaluations go through `Objective` (`value*`, `vmap_*`, or `value_function(...)` plus `log_evaluation(...)`; no direct `problem.objective_function()` calls)
- [ ] Objective `warmup_*()` happens before `obj.start_logging()`, while custom raw-callable compilation happens after it
- [ ] `random_seed` is accepted, set, and printed
- [ ] Early stopping uses `obj.evals_since_improvement` (or custom logic)
- [ ] Loop terminates on `obj.budget_exceeded`
- [ ] Imports added to `__init__.py` files
- [ ] Docstrings describe the algorithm, its hyperparameters, and provide a reference if applicable
