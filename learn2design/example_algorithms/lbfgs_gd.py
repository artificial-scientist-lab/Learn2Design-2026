# Reference: https://github.com/artificial-scientist-lab/Differometor-Benchmark/blob/main/src/dfbench/algorithms/gradient_based/lbfgs_gd.py
from __future__ import annotations

import math

import jax.numpy as jnp
import optax
from jaxtyping import Array, Float

from dfbench import Objective, OptimizationAlgorithm


class LBFGSGD(OptimizationAlgorithm):
    """Optax L-BFGS with Objective-managed Armijo evaluations.

    Every initial, iterate, and backtracking probe goes through
    Objective.value_and_grad(). No result is logged manually, so timing,
    feasibility, histories, and evaluation counts use the same mechanism as
    AdamGD.
    """

    algorithm_str = "lbfgs_gd"

    def __init__(self) -> None:
        pass

    def optimize(
        self,
        objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        patience: int | None = None,
        learning_rate: float = 1.0,
        memory_size: int = 10,
        scale_init_precond: bool = True,
        backtracking_factor: float = 0.5,
        armijo_coefficient: float = 1e-4,
        max_linesearch_steps: int = 20,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 < backtracking_factor < 1:
            raise ValueError("backtracking_factor must be between zero and one")
        if not 0 < armijo_coefficient < 1:
            raise ValueError("armijo_coefficient must be between zero and one")
        if max_linesearch_steps < 1:
            raise ValueError("max_linesearch_steps must be positive")

        obj = objective
        self.prepare(obj, unbounded=True, random_seed=random_seed)
        params = init_params if init_params is not None else obj.random_params_unbounded()

        # Optax provides the limited-memory inverse-Hessian direction. The
        # line search stays outside Optax so every probe can use the normal
        # Objective API and receive identical feasibility handling to Adam.
        optimizer = optax.lbfgs(
            learning_rate=1.0,
            memory_size=memory_size,
            scale_init_precond=scale_init_precond,
            linesearch=None,
        )
        opt_state = optimizer.init(params)

        obj.warmup_value_and_grad()
        obj.start_logging()
        loss, grads = obj.value_and_grad(params)

        while not obj.budget_exceeded:
            if patience is not None and obj.evals_since_improvement > patience:
                break

            updates, new_state = optimizer.update(
                grads,
                opt_state,
                params,
                value=loss,
                grad=grads,
            )
            slope = float(jnp.vdot(grads, updates))
            current_loss = float(loss)
            step_size = learning_rate

            candidate_params = params
            candidate_loss = loss
            candidate_grads = grads
            for _ in range(max_linesearch_steps):
                candidate_params = optax.apply_updates(
                    params,
                    step_size * updates,
                )
                candidate_loss, candidate_grads = obj.value_and_grad(candidate_params)
                if obj.budget_exceeded:
                    return

                candidate_value = float(candidate_loss)
                if math.isfinite(candidate_value):
                    if not math.isfinite(current_loss):
                        break
                    if slope < 0:
                        armijo_limit = (
                            current_loss
                            + armijo_coefficient * step_size * slope
                        )
                        if candidate_value <= armijo_limit:
                            break
                    elif candidate_value < current_loss:
                        break
                step_size *= backtracking_factor

            params = candidate_params
            loss = candidate_loss
            grads = candidate_grads
            opt_state = new_state
