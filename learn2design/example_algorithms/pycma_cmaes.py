"""Standalone vanilla CMA-ES backed by pycma.

Operates in bounded parameter space. The CMA search runs in the unit cube
[0, 1]^n and candidates are mapped to physical bounds at evaluation time.

Requires: pycma >= 3.3 (``uv add cma``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import jax.numpy as jnp

try:
    import cma
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pycma is required for PyCMACMAES. Install it with:  uv add cma"
    ) from exc

from dfbench import AlgorithmType, Objective, OptimizationAlgorithm


def _default_pop_size(n_params: int) -> int:
    """Return pycma's default population size for *n_params* dimensions."""
    return int(4 + 3 * np.floor(np.log(max(n_params, 1))))


def _build_opts(
    n: int,
    pop_size: int,
    active: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a pycma CMAOptions dict for unit-cube search.

    Searching in [0, 1]^n keeps a single scalar sigma meaningful across all
    coordinates, even when bound widths span orders of magnitude. Candidates
    are mapped to physical bounds only at evaluation time.
    """
    opts: dict[str, Any] = {
        "bounds": [[0.0] * n, [1.0] * n],
        "popsize": pop_size,
        "CMA_active": active,
        "verbose": -9,  # silent
        "maxfevals": np.inf,  # budget is managed by the Objective
        "tolx": 1e-12,
        "tolfun": 1e-12,
    }
    if extra:
        opts.update(extra)
    return opts


def _eval_batched(
    candidates: jnp.ndarray,
    obj: Objective,
    batch_size: int,
) -> jnp.ndarray:
    """Evaluate *candidates* through *obj* in batch_size-sized chunks."""
    n_pop = candidates.shape[0]
    if batch_size >= n_pop:
        return obj.vmap_value(candidates)  # single batched call
    all_losses = []
    for i in range(0, n_pop, batch_size):
        batch = candidates[i : i + batch_size]
        all_losses.append(obj.vmap_value(batch))
    return jnp.concatenate(all_losses, axis=0)


def _ask_eval_tell(
    es: "cma.CMAEvolutionStrategy",
    obj: Objective,
    lb_np: np.ndarray,
    width: np.ndarray,
    batch_size: int,
) -> None:
    """Run one ask-evaluate-tell generation in unit-cube coordinates.

    Candidates are clipped to [0, 1], mapped to physical space for
    evaluation, then the unit-cube solutions are passed back to es.tell so
    the covariance update stays in unit space.
    """
    solutions_unit = [np.clip(np.asarray(x), 0.0, 1.0) for x in es.ask()]
    physical = [lb_np + u * width for u in solutions_unit]
    batch = jnp.asarray(np.stack(physical))
    losses_jnp = _eval_batched(batch, obj, batch_size)
    es.tell(solutions_unit, np.asarray(losses_jnp).tolist())


class PyCMACMAES(OptimizationAlgorithm):
    """Vanilla CMA-ES backed by pycma (single run, no restarts)."""

    algorithm_str: str = "pycma_cmaes"
    algorithm_type: AlgorithmType = AlgorithmType.EVOLUTIONARY

    def __init__(self, batch_size: int = 1) -> None:
        self._batch_size = batch_size

    def optimize(
        self,
        objective: Objective,
        init_params: np.ndarray | None = None,
        random_seed: int | None = None,
        sigma0: float | None = None,
        pop_size: int | None = None,
        max_iterations: int | None = None,
    ) -> None:
        obj = objective

        random_seed, _ = self.prepare(obj, unbounded=False, random_seed=random_seed)

        bounds = np.asarray(obj.bounds)
        lb_np = np.asarray(bounds[0])
        ub_np = np.asarray(bounds[1])
        width = ub_np - lb_np
        n = obj.n_params

        # Build the initial mean in unit-cube coordinates.
        if init_params is None:
            x0_unit = np.random.uniform(0.0, 1.0, size=n)
        else:
            x0_phys = np.clip(np.asarray(init_params, dtype=float), lb_np, ub_np)
            x0_unit = (x0_phys - lb_np) / width

        # sigma is a dimensionless fraction of the unit cube.
        sigma = sigma0 if sigma0 is not None else 0.3
        pop_size = pop_size or _default_pop_size(n)
        opts = _build_opts(n, pop_size)

        # JIT warmup before timing starts.
        warmup_bs = min(self._batch_size, pop_size)
        obj.warmup_vmap_value(batch_size=warmup_bs)

        es = cma.CMAEvolutionStrategy(x0_unit.tolist(), sigma, opts)
        obj.start_logging()

        iteration = 0
        while not es.stop() and not obj.budget_exceeded:
            if max_iterations is not None and iteration >= max_iterations:
                break
            _ask_eval_tell(es, obj, lb_np, width, self._batch_size)
            iteration += 1
