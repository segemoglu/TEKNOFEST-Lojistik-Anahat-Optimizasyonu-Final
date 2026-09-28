"""Expose config-backed plan costing and independent validation entry points.

Callers pass domain ``ProblemData``/``TransportPlan`` models to the calculator
or validator; this package never loads Excel, generates routes or optimizes.
Minute/hour helpers raise explicit cost errors, while plan defects are returned
inside ``ValidationReport`` by ``TransportPlanValidator``.
"""

from .costs import (
    CostCalculationError,
    CostCalculator,
    ceil_billable_hours,
    ceil_billable_minutes,
)
from .policy import EvaluationPolicy, load_evaluation_policy
from .validator import TransportPlanValidator

__all__ = [
    "CostCalculationError",
    "CostCalculator",
    "EvaluationPolicy",
    "TransportPlanValidator",
    "ceil_billable_hours",
    "ceil_billable_minutes",
    "load_evaluation_policy",
]
