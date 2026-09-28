"""Exceptions for ALNS optimization module."""

from __future__ import annotations


class RepairInfeasibleError(Exception):
    """Raised when a repair operator cannot place an unassigned part into any feasible route."""
    pass


__all__ = ["RepairInfeasibleError"]
