"""Public API for deterministic, non-optimized baseline-plan strategies.

The package consumes standard ``ProblemData`` and ``EvaluationPolicy``
objects.  The conservative strategy carries one part per vehicle; the greedy
same-O-D strategy consolidates spot parts; the mandatory-first strategy fills
eligible fixed-route rentals before delegating remaining parts to spot FFD;
the two-stop strategy may combine two economical same-origin destinations.
The bounded multi-stop strategy may extend those seeds to three/four delivery
stops with real-cost cheapest insertion.  All hand plans to the existing cost
calculator and independent validator. This package does not load forecasts
itself, run a general optimizer or export a workbook.
"""

from .baseline import (
    BaselineBuildError,
    BaselineBuildResult,
    ConservativeDirectBaselineBuilder,
)
from .handling_ledger import HandlingCapacityLedger, HandlingUsage
from .partitioning import partition_demands
from .consolidated import GreedySameODConsolidatedBuilder
from .mandatory_filled import GreedyMandatoryRentalFilledBuilder
from .two_stop import GreedySameOriginTwoStopBuilder
from .multi_stop import GreedySameOriginMultiStopBuilder

__all__ = [
    "BaselineBuildError",
    "BaselineBuildResult",
    "ConservativeDirectBaselineBuilder",
    "GreedySameODConsolidatedBuilder",
    "GreedyMandatoryRentalFilledBuilder",
    "GreedySameOriginTwoStopBuilder",
    "GreedySameOriginMultiStopBuilder",
    "HandlingCapacityLedger",
    "HandlingUsage",
    "partition_demands",
]
