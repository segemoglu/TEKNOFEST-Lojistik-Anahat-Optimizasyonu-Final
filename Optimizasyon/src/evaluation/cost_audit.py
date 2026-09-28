"""Independently recompute candidate-plan costs for validator cross-checks.

This module is called only by :mod:`src.evaluation.validator`.  Its input is a
validated static/demand :class:`src.domain.ProblemData`, an
:class:`src.evaluation.policy.EvaluationPolicy` and a candidate
:class:`src.domain.solution_models.TransportPlan`; its output is a fresh
:class:`src.domain.solution_models.CostBreakdown`.  It deliberately does not
import :mod:`src.evaluation.costs`: keeping a second, small implementation of
the minute, kilometre (km), desi and Turkish-lira (TL) formulas prevents one
calculator defect from automatically being accepted by the validator.

Structural omissions such as an unknown vehicle type, missing matrix arc,
overlapping trips of one physical vehicle or reused mandatory-rental vehicle
identity raise :class:`CostAuditError`.  The validator converts those failures
to structured issues instead of exposing exceptions to callers.  The module
does not load files, build routes or optimize a plan.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from typing import Iterable

from src.domain import DemandInstance, ProblemData, TravelArc, VehicleType
from src.domain.solution_models import (
    CostBreakdown,
    SlaPenaltyLine,
    TransportPlan,
    VehicleCostLine,
    VehicleLeg,
    VehicleOwnershipType,
    VehicleTrip,
)

from .policy import EvaluationPolicy


SECONDS_PER_MINUTE = 60.0
MINUTES_PER_HOUR = 60.0


class CostAuditError(ValueError):
    """Signal structural plan data that prevents independent TL auditing.

    Internal audit helpers raise this for missing arcs/rates, invalid minutes,
    physical trip overlap or mandatory ID reuse.  ``TransportPlanValidator``
    catches it and returns ``independent_cost_audit_failed`` with context;
    ordinary plan callers therefore still receive a ``ValidationReport``.
    """


@dataclass(frozen=True, slots=True)
class _AuditCostUnit:
    """Hold one independently grouped unit to which hourly ``ceil`` applies."""

    cost_unit_id: str
    vehicle_id: str
    vehicle_type: str
    ownership_type: VehicleOwnershipType
    trips: tuple[VehicleTrip, ...]
    legs: tuple[VehicleLeg, ...]
    usage_minutes: float


def audit_cost_breakdown(
    problem: ProblemData,
    policy: EvaluationPolicy,
    plan: TransportPlan,
) -> CostBreakdown:
    """Recompute all vehicle and SLA costs without using ``CostCalculator``.

    Args:
        problem: Static network/rate data and original demand records.
        policy: Config-backed rounding, grouping and SLA-rate selections.
        plan: Candidate trips, handling operations and delivered demand parts.

    Returns:
        A line-level cost breakdown used by ``TransportPlanValidator`` as its
        independent expected result.

    Important rules:
        Vehicle time is calculated in minutes, then billed as
        ``ceil(minutes / 60 minutes/hour)``.  Distance uses km, SLA uses each
        part's desi and its own rounded delay hours, and all TL components are
        included even after 5 July.

    Raises:
        CostAuditError: If identities overlap/reuse, master data is missing,
        or an activity interval is negative or non-finite.
    """

    vehicle_types = {item.name: item for item in problem.static.vehicle_types}
    arcs = {
        (arc.origin, arc.destination): arc
        for arc in problem.static.travel_arcs
    }
    demands = {demand.demand_id: demand for demand in problem.demands}
    _assert_physical_vehicle_usage(plan.trips)

    units = _build_cost_units(plan, policy)
    vehicle_lines = tuple(
        _audit_vehicle_line(unit, vehicle_types, arcs, policy)
        for unit in units
    )
    sla_lines = tuple(_audit_sla_lines(plan, demands, arcs, policy))
    return _aggregate_breakdown(vehicle_lines, sla_lines)


def _assert_physical_vehicle_usage(trips: Iterable[VehicleTrip]) -> None:
    """Reject identity reuse that would hide or merge physical vehicle cost."""

    grouped: dict[str, list[VehicleTrip]] = defaultdict(list)
    for trip in trips:
        grouped[trip.vehicle_id].append(trip)

    for vehicle_id, vehicle_trips in grouped.items():
        identities = {
            (trip.vehicle_type, trip.ownership_type) for trip in vehicle_trips
        }
        if len(identities) != 1:
            raise CostAuditError(
                f"Aynı vehicle_id tür/sahiplik değiştirmiş: {vehicle_id!r}"
            )
        if (
            vehicle_trips
            and vehicle_trips[0].ownership_type == "mandatory_rented"
            and len(vehicle_trips) > 1
        ):
            raise CostAuditError(
                "Her günlük zorunlu kiralık ayrı vehicle_id kullanmalıdır: "
                f"{vehicle_id!r}"
            )

        ordered = sorted(vehicle_trips, key=lambda trip: _trip_bounds(trip)[0])
        for previous, following in zip(ordered, ordered[1:], strict=False):
            previous_end = _trip_bounds(previous)[1]
            following_start = _trip_bounds(following)[0]
            if following_start < previous_end:
                raise CostAuditError(
                    "Aynı fiziksel aracın trip zamanları çakışıyor: "
                    f"{vehicle_id!r}"
                )


def _build_cost_units(
    plan: TransportPlan,
    policy: EvaluationPolicy,
) -> tuple[_AuditCostUnit, ...]:
    """Group plan activity at vehicle, trip or leg rounding granularity."""

    if policy.hourly_cost_granularity == "vehicle":
        return _vehicle_units(plan.trips, policy)
    if policy.hourly_cost_granularity == "trip":
        return tuple(_trip_unit(trip) for trip in plan.trips)
    return tuple(
        _leg_unit(trip, leg)
        for trip in plan.trips
        for leg in trip.legs
    )


def _vehicle_units(
    trips: Iterable[VehicleTrip],
    policy: EvaluationPolicy,
) -> tuple[_AuditCostUnit, ...]:
    """Build one independent cost unit per physical vehicle identity."""

    grouped: dict[str, list[VehicleTrip]] = defaultdict(list)
    for trip in trips:
        grouped[trip.vehicle_id].append(trip)

    units: list[_AuditCostUnit] = []
    for vehicle_id, vehicle_trips in sorted(grouped.items()):
        ordered = tuple(sorted(vehicle_trips, key=lambda trip: _trip_bounds(trip)[0]))
        bounds = tuple(_trip_bounds(trip) for trip in ordered)
        if policy.include_idle_between_activities:
            usage_minutes = _duration_minutes(
                min(item[0] for item in bounds),
                max(item[1] for item in bounds),
            )
        else:
            usage_minutes = math.fsum(
                _duration_minutes(start_at, end_at)
                for start_at, end_at in bounds
            )
        representative = ordered[0]
        units.append(
            _AuditCostUnit(
                cost_unit_id=f"vehicle:{vehicle_id}",
                vehicle_id=vehicle_id,
                vehicle_type=representative.vehicle_type,
                ownership_type=representative.ownership_type,
                trips=ordered,
                legs=tuple(leg for trip in ordered for leg in trip.legs),
                usage_minutes=usage_minutes,
            )
        )
    return tuple(units)


def _trip_unit(trip: VehicleTrip) -> _AuditCostUnit:
    """Build one independent cost unit for a declared trip."""

    return _AuditCostUnit(
        cost_unit_id=f"trip:{trip.trip_id}",
        vehicle_id=trip.vehicle_id,
        vehicle_type=trip.vehicle_type,
        ownership_type=trip.ownership_type,
        trips=(trip,),
        legs=trip.legs,
        usage_minutes=_duration_minutes(*_trip_bounds(trip)),
    )


def _leg_unit(trip: VehicleTrip, leg: VehicleLeg) -> _AuditCostUnit:
    """Build one independent cost unit for a leg and its linked handling."""

    return _AuditCostUnit(
        cost_unit_id=f"leg:{leg.leg_id}",
        vehicle_id=trip.vehicle_id,
        vehicle_type=trip.vehicle_type,
        ownership_type=trip.ownership_type,
        trips=(trip,),
        legs=(leg,),
        usage_minutes=_duration_minutes(*_leg_bounds(trip, leg)),
    )


def _audit_vehicle_line(
    unit: _AuditCostUnit,
    vehicle_types: dict[str, VehicleType],
    arcs: dict[tuple[str, str], TravelArc],
    policy: EvaluationPolicy,
) -> VehicleCostLine:
    """Independently apply minute/hour and km/TL formulas to one unit."""

    try:
        vehicle_type = vehicle_types[unit.vehicle_type]
    except KeyError as exc:
        raise CostAuditError(f"Bilinmeyen araç türü: {unit.vehicle_type!r}") from exc
    hourly_rate, distance_rate = _rates(vehicle_type, unit.ownership_type)
    distance_km = math.fsum(_arc(arcs, leg).distance_km for leg in unit.legs)
    charged_hours = _ceil_hours_from_minutes(unit.usage_minutes)
    hourly_cost_tl = charged_hours * hourly_rate
    distance_cost_tl = distance_km * distance_rate
    return VehicleCostLine(
        cost_unit_id=unit.cost_unit_id,
        granularity=policy.hourly_cost_granularity,
        vehicle_id=unit.vehicle_id,
        ownership_type=unit.ownership_type,
        trip_ids=tuple(trip.trip_id for trip in unit.trips),
        leg_ids=tuple(leg.leg_id for leg in unit.legs),
        usage_hours=unit.usage_minutes / MINUTES_PER_HOUR,
        charged_hours=charged_hours,
        distance_km=distance_km,
        hourly_cost_tl=hourly_cost_tl,
        distance_cost_tl=distance_cost_tl,
        total_cost_tl=hourly_cost_tl + distance_cost_tl,
    )


def _audit_sla_lines(
    plan: TransportPlan,
    demands: dict[str, DemandInstance],
    arcs: dict[tuple[str, str], TravelArc],
    policy: EvaluationPolicy,
) -> list[SlaPenaltyLine]:
    """Independently calculate each demand part's final-delivery SLA line."""

    parts = {part.part_id: part for part in plan.demand_parts}
    lines: list[SlaPenaltyLine] = []
    for trip in plan.trips:
        operations = {item.operation_id: item for item in trip.handling_operations}
        for leg in trip.legs:
            for assignment in leg.assignments:
                if not assignment.is_final_delivery:
                    continue
                part = parts.get(assignment.demand_part_id)
                if part is None or part.demand_id not in demands:
                    raise CostAuditError(
                        "SLA auditinde bilinmeyen demand part veya demand_id: "
                        f"{assignment.demand_part_id!r}"
                    )
                demand = demands[part.demand_id]
                unload = operations.get(leg.unloading_operation_id or "")
                delivered_at = unload.end_at if unload is not None else leg.arrival_at
                demand_arc = _arc_by_key(arcs, demand.origin, demand.destination)
                due_at = datetime.combine(
                    demand.demand_date,
                    demand.time_slot,
                ) + timedelta(days=demand_arc.target_delivery_days)
                # Early delivery is a zero-minute delay, not a negative
                # activity interval error.
                delay_minutes = max(
                    0.0,
                    (delivered_at - due_at).total_seconds()
                    / SECONDS_PER_MINUTE,
                )
                charged_delay_hours = _ceil_hours_from_minutes(delay_minutes)
                penalty_tl = (
                    assignment.desi
                    * charged_delay_hours
                    * policy.sla_penalty_per_desi_hour_tl
                )
                lines.append(
                    SlaPenaltyLine(
                        demand_id=demand.demand_id,
                        demand_part_id=part.part_id,
                        delivered_desi=assignment.desi,
                        due_at=due_at,
                        delivered_at=delivered_at,
                        delay_hours=delay_minutes / MINUTES_PER_HOUR,
                        charged_delay_hours=charged_delay_hours,
                        penalty_tl=penalty_tl,
                    )
                )
    return lines


def _aggregate_breakdown(
    vehicle_lines: tuple[VehicleCostLine, ...],
    sla_lines: tuple[SlaPenaltyLine, ...],
) -> CostBreakdown:
    """Aggregate independently audited line items into TL components."""

    mandatory_hourly = math.fsum(
        line.hourly_cost_tl
        for line in vehicle_lines
        if line.ownership_type == "mandatory_rented"
    )
    mandatory_distance = math.fsum(
        line.distance_cost_tl
        for line in vehicle_lines
        if line.ownership_type == "mandatory_rented"
    )
    spot_hourly = math.fsum(
        line.hourly_cost_tl
        for line in vehicle_lines
        if line.ownership_type == "spot"
    )
    spot_distance = math.fsum(
        line.distance_cost_tl
        for line in vehicle_lines
        if line.ownership_type == "spot"
    )
    sla_penalty = math.fsum(line.penalty_tl for line in sla_lines)
    total = mandatory_hourly + mandatory_distance + spot_hourly + spot_distance
    total += sla_penalty
    return CostBreakdown(
        vehicle_cost_lines=vehicle_lines,
        sla_penalty_lines=sla_lines,
        mandatory_rented_hourly_cost_tl=mandatory_hourly,
        mandatory_rented_distance_cost_tl=mandatory_distance,
        spot_hourly_cost_tl=spot_hourly,
        spot_distance_cost_tl=spot_distance,
        sla_penalty_tl=sla_penalty,
        total_cost_tl=total,
    )


def _trip_bounds(trip: VehicleTrip) -> tuple[datetime, datetime]:
    """Return the activity envelope including handling, travel and waiting."""

    starts = [leg.departure_at for leg in trip.legs]
    ends = [leg.arrival_at for leg in trip.legs]
    starts.extend(item.start_at for item in trip.handling_operations)
    ends.extend(item.end_at for item in trip.handling_operations)
    if not starts or not ends:
        raise CostAuditError(f"Maliyetlenecek trip aktivitesi yok: {trip.trip_id!r}")
    return min(starts), max(ends)


def _leg_bounds(trip: VehicleTrip, leg: VehicleLeg) -> tuple[datetime, datetime]:
    """Return one leg's travel envelope extended by linked handling."""

    operations = {item.operation_id: item for item in trip.handling_operations}
    start_at = leg.departure_at
    end_at = leg.arrival_at
    if leg.loading_operation_id in operations:
        start_at = min(start_at, operations[leg.loading_operation_id].start_at)
    if leg.unloading_operation_id in operations:
        end_at = max(end_at, operations[leg.unloading_operation_id].end_at)
    return start_at, end_at


def _duration_minutes(start_at: datetime, end_at: datetime) -> float:
    """Convert one datetime interval to minutes and reject negative values."""

    minutes = (end_at - start_at).total_seconds() / SECONDS_PER_MINUTE
    if not math.isfinite(minutes) or minutes < 0:
        raise CostAuditError("Aktivite süresi sonlu ve negatif olmayan dakika olmalıdır.")
    return minutes


def _ceil_hours_from_minutes(minutes: float) -> int:
    """Apply ``ceil(minutes / 60 minutes/hour)`` independently."""

    if not math.isfinite(minutes) or minutes < 0:
        raise CostAuditError("Ücretlendirilen süre sonlu ve negatif olamaz.")
    return math.ceil(minutes / MINUTES_PER_HOUR)


def _rates(
    vehicle_type: VehicleType,
    ownership_type: VehicleOwnershipType,
) -> tuple[float, float]:
    """Select hourly TL/hour and distance TL/km rates by ownership."""

    if ownership_type == "mandatory_rented":
        return (
            vehicle_type.rented_hourly_cost_tl,
            vehicle_type.rented_cost_per_km_tl,
        )
    if ownership_type == "spot":
        return vehicle_type.spot_hourly_cost_tl, vehicle_type.spot_cost_per_km_tl
    raise CostAuditError(f"Bilinmeyen araç sahipliği: {ownership_type!r}")


def _arc(
    arcs: dict[tuple[str, str], TravelArc],
    leg: VehicleLeg,
) -> TravelArc:
    """Resolve the directed matrix arc used for one leg's kilometre cost."""

    return _arc_by_key(arcs, leg.origin, leg.destination)


def _arc_by_key(
    arcs: dict[tuple[str, str], TravelArc],
    origin: str,
    destination: str,
) -> TravelArc:
    """Resolve a directed arc or explain why independent audit cannot finish."""

    try:
        return arcs[(origin, destination)]
    except KeyError as exc:
        raise CostAuditError(
            f"Mesafe/seyahat yayı bulunamadı: {origin!r} -> {destination!r}"
        ) from exc


__all__ = ["CostAuditError", "audit_cost_breakdown"]
