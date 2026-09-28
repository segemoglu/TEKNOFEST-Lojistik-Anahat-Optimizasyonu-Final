"""Source-independent dataclasses that describe a candidate transport plan.

The input models in :mod:`src.domain.models` describe what must be solved.
This module describes a proposed solution and the structured results produced
when that proposal is costed and validated.  It intentionally contains no
optimizer, route generator or Excel-export behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from types import MappingProxyType
from typing import Any, Literal, Mapping


# These literal types keep ownership and handling values explicit throughout
# costing and validation without introducing framework-specific enums.
VehicleOwnershipType = Literal["mandatory_rented", "spot"]
HandlingOperationType = Literal["load", "unload"]
CostGranularity = Literal["vehicle", "trip", "leg"]


@dataclass(frozen=True, slots=True)
class DemandPart:
    """An atomic, traceable desi quantity from one original demand.

    ``demand_id`` links this input to
    :class:`~src.domain.models.DemandInstance`; validator and costing code use
    ``part_id`` for end-to-end identity.  Every leg assignment must carry this
    exact ``desi`` amount.  Further splitting requires new hierarchical child
    IDs such as ``D0001-1-1`` and ``D0001-1-2`` instead of partial reuse of the
    parent ID.  Invalid balances are returned as structured validator issues.
    """

    part_id: str
    demand_id: str
    desi: float


@dataclass(frozen=True, slots=True)
class DemandAssignment:
    """Place one complete atomic demand part on one physical vehicle leg.

    ``demand_part_id`` and ``desi`` must exactly match a ``DemandPart``.  The
    load/unload flags connect this occurrence to handling operations: a false
    load flag is allowed only when the same part remains on the immediately
    preceding leg of the same physical vehicle.  ``is_final_delivery`` marks
    the single unload at the original destination.  The validator reports
    partial reuse, cargo creation/loss, invalid transfer or duplicate finals.
    """

    demand_part_id: str
    desi: float
    loaded_at_origin: bool = True
    unloaded_at_destination: bool = True
    is_final_delivery: bool = False


@dataclass(frozen=True, slots=True)
class VehicleLeg:
    """One physical movement between two transfer centres.

    Loading and unloading are represented by separate
    :class:`HandlingOperation` objects.  Their IDs are optional for empty
    mandatory movements, while the validator requires them whenever positive
    desi is loaded or unloaded.  Validator, costing and independent audit code
    consume the minute timestamps; invalid chronology, matrix duration,
    capacity or cargo continuity becomes a structured validation issue.
    """

    leg_id: str
    origin: str
    destination: str
    departure_at: datetime
    arrival_at: datetime
    assignments: tuple[DemandAssignment, ...] = ()
    loading_operation_id: str | None = None
    unloading_operation_id: str | None = None

    def __post_init__(self) -> None:
        """Freeze assignment collections supplied as lists by callers."""

        object.__setattr__(self, "assignments", tuple(self.assignments))


@dataclass(frozen=True, slots=True)
class HandlingOperation:
    """A loading or unloading activity performed at one transfer centre.

    Overlap with another vehicle's handling operation is deliberately legal.
    Daily capacity is validated by summing handled desi, not by applying a
    ramp, queue or ``NoOverlap`` constraint.
    """

    operation_id: str
    vehicle_id: str
    center: str
    operation_type: HandlingOperationType
    start_at: datetime
    end_at: datetime
    desi: float
    demand_part_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Freeze the list of demand parts touched by this operation."""

        object.__setattr__(self, "demand_part_ids", tuple(self.demand_part_ids))


@dataclass(frozen=True, slots=True)
class VehicleTrip:
    """A continuous trip made by one physical vehicle.

    Multiple trips may share ``vehicle_id``.  The next trip must start at the
    previous trip's final centre unless an explicit empty repositioning leg
    moves the vehicle.  This identity also prevents cost from being multiplied
    by the number of demands carried and enables config-driven vehicle-, trip-
    or leg-level hourly rounding.  Validator and cost modules consume the
    immutable leg/handling collections and report inconsistent identities.
    """

    trip_id: str
    vehicle_id: str
    vehicle_type: str
    ownership_type: VehicleOwnershipType
    legs: tuple[VehicleLeg, ...]
    handling_operations: tuple[HandlingOperation, ...] = ()

    def __post_init__(self) -> None:
        """Freeze leg and handling collections supplied by plan builders."""

        object.__setattr__(self, "legs", tuple(self.legs))
        object.__setattr__(
            self,
            "handling_operations",
            tuple(self.handling_operations),
        )


@dataclass(frozen=True, slots=True)
class VehicleCostLine:
    """One auditable vehicle cost unit produced by the cost calculator."""

    cost_unit_id: str
    granularity: CostGranularity
    vehicle_id: str
    ownership_type: VehicleOwnershipType
    trip_ids: tuple[str, ...]
    leg_ids: tuple[str, ...]
    usage_hours: float
    charged_hours: int
    distance_km: float
    hourly_cost_tl: float
    distance_cost_tl: float
    total_cost_tl: float

    def __post_init__(self) -> None:
        """Freeze traceability IDs so a reported line cannot mutate later."""

        object.__setattr__(self, "trip_ids", tuple(self.trip_ids))
        object.__setattr__(self, "leg_ids", tuple(self.leg_ids))


@dataclass(frozen=True, slots=True)
class SlaPenaltyLine:
    """One final-delivery SLA calculation for a demand part."""

    demand_id: str
    demand_part_id: str
    delivered_desi: float
    due_at: datetime
    delivered_at: datetime
    delay_hours: float
    charged_delay_hours: int
    penalty_tl: float


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Vehicle and SLA costs with both totals and auditable line items."""

    vehicle_cost_lines: tuple[VehicleCostLine, ...] = ()
    sla_penalty_lines: tuple[SlaPenaltyLine, ...] = ()
    mandatory_rented_hourly_cost_tl: float = 0.0
    mandatory_rented_distance_cost_tl: float = 0.0
    spot_hourly_cost_tl: float = 0.0
    spot_distance_cost_tl: float = 0.0
    sla_penalty_tl: float = 0.0
    total_cost_tl: float = 0.0

    def __post_init__(self) -> None:
        """Freeze cost line collections used by the independent validator."""

        object.__setattr__(
            self,
            "vehicle_cost_lines",
            tuple(self.vehicle_cost_lines),
        )
        object.__setattr__(
            self,
            "sla_penalty_lines",
            tuple(self.sla_penalty_lines),
        )

    @property
    def vehicle_cost_tl(self) -> float:
        """Return all hourly and distance-based vehicle costs."""

        return (
            self.mandatory_rented_hourly_cost_tl
            + self.mandatory_rented_distance_cost_tl
            + self.spot_hourly_cost_tl
            + self.spot_distance_cost_tl
        )


@dataclass(frozen=True, slots=True)
class TransportPlan:
    """A complete candidate solution evaluated without running an optimizer."""

    demand_parts: tuple[DemandPart, ...]
    trips: tuple[VehicleTrip, ...]
    reported_cost: CostBreakdown | None = None

    def __post_init__(self) -> None:
        """Freeze top-level plan collections for repeatable validation."""

        object.__setattr__(self, "demand_parts", tuple(self.demand_parts))
        object.__setattr__(self, "trips", tuple(self.trips))


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One plan problem with demand, vehicle, date and value context."""

    code: str
    message: str
    demand_id: str | None = None
    vehicle_id: str | None = None
    event_date: date | None = None
    trip_id: str | None = None
    leg_id: str | None = None
    related_values: Mapping[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        """Protect issue context from mutation after a report is produced."""

        object.__setattr__(
            self,
            "related_values",
            MappingProxyType(dict(self.related_values)),
        )


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Structured validation outcome returned instead of raising plan errors."""

    is_valid: bool
    errors: tuple[ValidationIssue, ...] = ()
    warnings: tuple[ValidationIssue, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        """Freeze report content and keep ``is_valid`` consistent with errors."""

        object.__setattr__(self, "errors", tuple(self.errors))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "is_valid", not self.errors)


__all__ = [
    "CostBreakdown",
    "CostGranularity",
    "DemandAssignment",
    "DemandPart",
    "HandlingOperation",
    "HandlingOperationType",
    "SlaPenaltyLine",
    "TransportPlan",
    "ValidationIssue",
    "ValidationReport",
    "VehicleCostLine",
    "VehicleLeg",
    "VehicleOwnershipType",
    "VehicleTrip",
]
