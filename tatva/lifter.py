from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Self

import equinox
import jax.numpy as jnp
import numpy as np
from jax import Array


class Constraint(Hashable):
    """Abstract base class for conditions applied during lifting.

    Subclasses define how constrained degrees of freedom (dofs) are enforced
    when mapping a reduced vector back to the full vector.
    """

    dofs: Array
    """The constrained dofs (an array of integer indices)."""

    def apply_lift(self, u_full: Array) -> Array:  # override in subclasses
        """Apply the constraint to a full vector and return the modified copy."""
        return u_full


@dataclass
class PeriodicMap(Constraint):
    dofs: Array
    master_dofs: Array

    def apply_lift(self, u_full: Array) -> Array:
        """Copy values from ``master_dofs`` into the constrained ``dofs``."""
        return u_full.at[self.dofs].set(u_full[self.master_dofs])


@dataclass
class DirichletBC(Constraint):
    dofs: Array
    values: Array

    def apply_lift(self, u_full: Array) -> Array:
        """Set constrained ``dofs`` to fixed ``values``."""
        return u_full.at[self.dofs].set(self.values)


class Lifter(equinox.Module):
    """Create a lifter that maps between reduced and full vectors.

    Args:
        size: Total number of dofs in the full vector.
        dirichlet_dofs: Dofs fixed to be 0 by Dirichlet boundary conditions.
        additional_constraints: Extra constraints (e.g., periodic maps).
        **kwargs: Ignored; kept for compatibility with equinox.Module init.

    Examples::

        lifter = Lifter(
            6,
            jnp.array([0, 5]),
            PeriodicMap(dofs=jnp.array([2]), master_dofs=jnp.array([1])),
        )
        u_reduced = jnp.array([10.0, 20.0])
        u_full = lifter.lift_from_null(u_reduced)
        # u_full -> [0., 10., 10., 20., 0., 0.]
        u_reduced_back = lifter.reduce(u_full)

    """

    free_dofs: Array
    """Array of free dofs as integer indices (not constrained)."""

    constrained_dofs: Array
    """Array of constrained dofs as integer indices."""

    size: int
    """Total number of dofs in the full vector."""

    size_reduced: int
    """Number of dofs in the reduced vector (free dofs only)."""

    constraints: tuple[Constraint, ...] = ()
    """Tuple of additional constraints (e.g., periodic maps)."""

    def __init__(
        self,
        size: int,
        dirichlet_dofs: Array,
        /,
        *additional_constraints: Constraint,
        **kwargs,
    ):
        self.size = size
        self.constraints = additional_constraints

        self._compute_sizes(dirichlet_dofs)

    def __hash__(self):
        return hash(self.constraints)

    def _compute_sizes(self, dirichlet_dofs: Array):
        """Compute free/constrained dofs and reduced size."""
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
        """Return a new lifter with ``condition`` appended to constraints."""
        return equinox.tree_at(
            lambda lf: lf.constraints, self, self.constraints + (condition,)
        )

    def lift(self, u_reduced: Array, u_full: Array) -> Array:
        """Lift reduced displacement vector to full size.

        Args:
            u_reduced: Vector on free dofs (length ``size_reduced``).
            u_full: Base full vector to modify; typically previous solution.

        Returns:
            Full vector with free dofs set to ``u_reduced`` and constraints
            applied (Dirichlet, periodic, etc.).
        """
        assert u_reduced.shape[0] == self.size_reduced, (
            f"Reduced displacement vector has incorrect size: "
            f"expected {self.size_reduced}, got {u_reduced.shape[0]}"
        )
        u_full = u_full.at[self.free_dofs].set(u_reduced)
        for condition in self.constraints:
            u_full = condition.apply_lift(u_full)
        return u_full

    def lift_from_null(self, u_reduced: Array) -> Array:
        """Lift reduced vector to a full vector starting from zeros."""
        u_full = jnp.zeros(self.size, dtype=u_reduced.dtype)
        return self.lift(u_reduced, u_full)

    def reduce(self, u_full: Array) -> Array:
        """Extract the reduced vector by selecting free dofs from ``u_full``."""
        return u_full[self.free_dofs]
