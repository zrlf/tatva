from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Self

import equinox
import jax.numpy as jnp
import numpy as np
from jax import Array


class Constraint(Hashable):
    """Abstract base class for conditions applied during lifting."""

    dofs: Array
    """The constrained dofs"""

    def apply_lift(self, u_full: Array) -> Array:  # override in subclasses
        return u_full


@dataclass
class PeriodicMap(Constraint):
    dofs: Array
    master_dofs: Array

    def apply_lift(self, u_full: Array) -> Array:
        return u_full.at[self.dofs].set(u_full[self.master_dofs])


@dataclass
class DirichletBC(Constraint):
    dofs: Array
    values: Array

    def apply_lift(self, u_full: Array) -> Array:
        return u_full.at[self.dofs].set(self.values)


class Lifter(equinox.Module):
    free_dofs: Array
    constrained_dofs: Array
    size: int
    size_reduced: int
    constraints: tuple[Constraint, ...] = ()

    def __init__(
        self,
        size: int,
        dirichlet_dofs: Array,
        *additional_constraints: Constraint,
        **kwargs,
    ):
        self.size = size
        self.constraints = additional_constraints

        self._compute_sizes(dirichlet_dofs)

    def __hash__(self):
        return hash(self.constraints)

    def _compute_sizes(self, dirichlet_dofs: Array):
        all_dofs = jnp.arange(self.size)
        constrained = jnp.concatenate(
            [dirichlet_dofs] + [cond.dofs for cond in self.constraints]
        )
        constrained = jnp.unique(constrained)
        free = jnp.setdiff1d(all_dofs, constrained, assume_unique=True)

        self.free_dofs = free
        self.constrained_dofs = constrained
        self.size_reduced = free.size

    def add(self, condition: Constraint) -> Self:
        """Add a condition to the lifter."""
        return equinox.tree_at(
            lambda lf: lf.constraints, self, self.constraints + (condition,)
        )

    def lift(self, u_reduced: Array, u_full: Array) -> Array:
        """Lift reduced displacement vector to full size."""
        assert u_reduced.shape[0] == self.size_reduced, (
            f"Reduced displacement vector has incorrect size: "
            f"expected {self.size_reduced}, got {u_reduced.shape[0]}"
        )
        u_full = u_full.at[self.free_dofs].set(u_reduced)
        for condition in self.constraints:
            u_full = condition.apply_lift(u_full)
        return u_full

    def lift_from_null(self, u_reduced: Array) -> Array:
        """Lift reduced displacement vector to full size from zero."""
        u_full = jnp.zeros(self.size, dtype=u_reduced.dtype)
        return self.lift(u_reduced, u_full)

    def reduce(self, u_full: Array) -> Array:
        """Reduce full displacement vector to free dofs."""
        return u_full[self.free_dofs]
