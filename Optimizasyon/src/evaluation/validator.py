"""Independent, non-throwing validation for candidate transport plans.

The validator checks a plan against :class:`src.domain.ProblemData` and a
config-backed :class:`src.evaluation.policy.EvaluationPolicy`.  Expected plan
problems are accumulated into :class:`src.domain.ValidationReport`; they are
never raised as exceptions.  No route construction or optimization occurs in
this module.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import math
from typing import Any

from src.domain import (
    MandatoryRentalRoute,
    ProblemData,
    ValidationIssue,
    ValidationReport,
)
from src.domain.solution_models import (
    CostBreakdown,
    DemandAssignment,
    DemandPart,
    HandlingOperation,
    SlaPenaltyLine,
    TransportPlan,
    VehicleCostLine,
    VehicleLeg,
    VehicleTrip,
)

from .cost_audit import CostAuditError, audit_cost_breakdown
from .policy import EvaluationPolicy


SECONDS_PER_MINUTE = 60.0
MINUTES_PER_HOUR = 60.0
SECONDS_PER_HOUR = SECONDS_PER_MINUTE * MINUTES_PER_HOUR


@dataclass(frozen=True, slots=True)
class _PlanIndex:
    """Lookup tables built once and shared by all validation checks."""

    parts: dict[str, DemandPart]
    operations: dict[str, HandlingOperation]
    trip_by_operation: dict[str, VehicleTrip]


@dataclass(frozen=True, slots=True)
class _FinalDelivery:
    """Normalized final-delivery event used by demand and schedule checks."""

    demand_id: str
    part_id: str
    vehicle_id: str
    trip_id: str
    leg_id: str
    destination: str
    desi: float
    delivered_at: datetime


@dataclass(frozen=True, slots=True)
class _CargoOccurrence:
    """Bind one atomic demand-part assignment to its physical leg context."""

    trip: VehicleTrip
    leg: VehicleLeg
    assignment: DemandAssignment
    load_operation: HandlingOperation | None
    unload_operation: HandlingOperation | None


class TransportPlanValidator:
    """Validate a complete plan and return a structured non-throwing report.

    Callers provide source-independent ``ProblemData``, ``EvaluationPolicy``
    and ``TransportPlan`` instances.  The validator checks atomic cargo flow,
    transfer and physical vehicle location continuity, demand/desi balance,
    chronology, capacity, handling, daily Tır visits, mandatory schedules, SLA
    and independently recomputed TL costs.  Expected plan defects become
    ``ValidationIssue`` records; malformed evaluator setup may still raise
    during construction.
    """

    def __init__(self, problem: ProblemData, policy: EvaluationPolicy) -> None:
        """Index validated static inputs without coupling to Excel loading."""

        self.problem = problem
        self.policy = policy
        self._demands = {
            demand.demand_id: demand for demand in problem.demands
        }
        self._vehicle_types = {
            vehicle.name: vehicle for vehicle in problem.static.vehicle_types
        }
        self._arcs = {
            (arc.origin, arc.destination): arc
            for arc in problem.static.travel_arcs
        }
        self._handling_capacities = {
            capacity.center: capacity.max_daily_desi
            for capacity in problem.static.handling_capacities
        }
        self._truck_capacities = {
            capacity.center: capacity.max_daily_truck_visits
            for capacity in problem.static.daily_truck_capacities
        }

    def validate(self, plan: TransportPlan) -> ValidationReport:
        """Run all business checks and return a structured validation report.

        Args:
            plan: Candidate demand parts, vehicle trips, handling operations and
                optional reported TL breakdown.

        Returns:
            ``ValidationReport`` containing errors, warnings and metrics for
            demand IDs, vehicle IDs, dates, desi, minutes, visits and TL totals.

        Connected modules and errors:
            Callers construct the plan with ``domain.solution_models``.  This
            method invokes ``cost_audit`` rather than ``CostCalculator`` for an
            independent expected cost.  Expected plan defects never escape as
            exceptions; structural audit failures become validation issues.
        """

        errors: list[ValidationIssue] = []
        warnings: list[ValidationIssue] = []
        metrics = self._initial_metrics(plan)

        # Indexing reports duplicate identifiers but retains the first object,
        # allowing all remaining checks to continue and reveal more problems.
        index, index_errors = self._build_index(plan)
        errors.extend(index_errors)
        errors.extend(self._check_non_negative_values(plan))
        errors.extend(self._check_vehicle_identity(plan))
        errors.extend(self._check_assignments_and_capacity(plan, index))
        errors.extend(self._check_time_sequence_and_matrix(plan, index))
        errors.extend(self._check_cargo_flow(plan, index))

        demand_errors, deliveries, demand_metrics = self._check_demand_delivery(
            plan,
            index,
        )
        errors.extend(demand_errors)
        metrics.update(demand_metrics)

        handling_errors, handling_metrics = self._check_handling_capacity(plan)
        errors.extend(handling_errors)
        metrics.update(handling_metrics)

        truck_errors, truck_metrics = self._check_daily_truck_quota(plan)
        errors.extend(truck_errors)
        metrics.update(truck_metrics)

        mandatory_errors, mandatory_metrics = self._check_mandatory_schedule(
            plan,
            deliveries,
            demand_metrics["all_demands_fully_delivered"],
        )
        errors.extend(mandatory_errors)
        metrics.update(mandatory_metrics)

        cost_errors, cost_warnings, cost_metrics = self._audit_plan_cost(plan)
        errors.extend(cost_errors)
        warnings.extend(cost_warnings)
        metrics.update(cost_metrics)

        metrics["error_count"] = len(errors)
        metrics["warning_count"] = len(warnings)
        return ValidationReport(
            is_valid=not errors,
            errors=tuple(errors),
            warnings=tuple(warnings),
            metrics=metrics,
        )

    def _initial_metrics(self, plan: TransportPlan) -> dict[str, Any]:
        """Build deterministic plan-size metrics before business checks run."""

        return {
            "demand_count": len(self.problem.demands),
            "demand_part_count": len(plan.demand_parts),
            "trip_count": len(plan.trips),
            "vehicle_count": len({trip.vehicle_id for trip in plan.trips}),
            "leg_count": sum(len(trip.legs) for trip in plan.trips),
            "handling_operation_count": sum(
                len(trip.handling_operations) for trip in plan.trips
            ),
        }

    def _audit_plan_cost(
        self,
        plan: TransportPlan,
    ) -> tuple[list[ValidationIssue], list[ValidationIssue], dict[str, Any]]:
        """Run independent TL audit and convert structural failures to issues."""

        try:
            computed = audit_cost_breakdown(self.problem, self.policy, plan)
        except (CostAuditError, KeyError, ValueError) as exc:
            return [
                self._issue(
                    "independent_cost_audit_failed",
                    f"Bağımsız maliyet denetimi tamamlanamadı: {exc}",
                    related_values={"exception_type": type(exc).__name__},
                )
            ], [], {}
        return self._check_cost_breakdown(plan.reported_cost, computed)

    def _build_index(
        self,
        plan: TransportPlan,
    ) -> tuple[_PlanIndex, list[ValidationIssue]]:
        """Index plan IDs and report every duplicate identifier."""

        parts, errors = self._index_demand_parts(plan.demand_parts)
        trips: dict[str, VehicleTrip] = {}
        legs: dict[str, VehicleLeg] = {}
        operations: dict[str, HandlingOperation] = {}
        trip_by_operation: dict[str, VehicleTrip] = {}
        for trip in plan.trips:
            errors.extend(self._index_trip(trip, trips))
            errors.extend(self._index_legs(trip, legs))
            errors.extend(
                self._index_operations(trip, operations, trip_by_operation)
            )

        return (
            _PlanIndex(
                parts=parts,
                operations=operations,
                trip_by_operation=trip_by_operation,
            ),
            errors,
        )

    def _index_demand_parts(
        self,
        parts_to_index: tuple[DemandPart, ...],
    ) -> tuple[dict[str, DemandPart], list[ValidationIssue]]:
        """Index demand parts while retaining the first duplicate ID."""

        parts: dict[str, DemandPart] = {}
        errors: list[ValidationIssue] = []
        for part in parts_to_index:
            if part.part_id in parts:
                errors.append(
                    self._issue(
                        "duplicate_demand_part_id",
                        "Demand part kimliği birden fazla kez kullanılmış.",
                        demand_id=part.demand_id,
                        related_values={"part_id": part.part_id},
                    )
                )
            else:
                parts[part.part_id] = part
        return parts, errors

    def _index_trip(
        self,
        trip: VehicleTrip,
        trips: dict[str, VehicleTrip],
    ) -> list[ValidationIssue]:
        """Index one trip and return a duplicate-ID issue when necessary."""

        if trip.trip_id not in trips:
            trips[trip.trip_id] = trip
            return []
        return [
            self._issue(
                "duplicate_trip_id",
                "Trip kimliği birden fazla kez kullanılmış.",
                vehicle_id=trip.vehicle_id,
                trip_id=trip.trip_id,
                related_values={"trip_id": trip.trip_id},
            )
        ]

    def _index_legs(
        self,
        trip: VehicleTrip,
        legs: dict[str, VehicleLeg],
    ) -> list[ValidationIssue]:
        """Index a trip's legs and report duplicate global leg IDs."""

        errors: list[ValidationIssue] = []
        for leg in trip.legs:
            if leg.leg_id not in legs:
                legs[leg.leg_id] = leg
                continue
            errors.append(
                self._issue(
                    "duplicate_leg_id",
                    "Araç bacağı kimliği birden fazla kez kullanılmış.",
                    vehicle_id=trip.vehicle_id,
                    event_date=leg.departure_at.date(),
                    trip_id=trip.trip_id,
                    leg_id=leg.leg_id,
                    related_values={"leg_id": leg.leg_id},
                )
            )
        return errors

    def _index_operations(
        self,
        trip: VehicleTrip,
        operations: dict[str, HandlingOperation],
        trip_by_operation: dict[str, VehicleTrip],
    ) -> list[ValidationIssue]:
        """Index handling operations and report duplicate global IDs."""

        errors: list[ValidationIssue] = []
        for operation in trip.handling_operations:
            if operation.operation_id not in operations:
                operations[operation.operation_id] = operation
                trip_by_operation[operation.operation_id] = trip
                continue
            errors.append(
                self._issue(
                    "duplicate_handling_operation_id",
                    "Elleçleme kimliği birden fazla kez kullanılmış.",
                    vehicle_id=operation.vehicle_id,
                    event_date=operation.start_at.date(),
                    trip_id=trip.trip_id,
                    related_values={"operation_id": operation.operation_id},
                )
            )
        return errors

    def _check_non_negative_values(
        self,
        plan: TransportPlan,
    ) -> list[ValidationIssue]:
        """Report negative desi, duration and reported cost values."""

        errors = self._check_negative_parts(plan.demand_parts)
        parts = {part.part_id: part for part in plan.demand_parts}
        for trip in plan.trips:
            for leg in trip.legs:
                errors.extend(self._check_negative_leg_values(trip, leg, parts))
            for operation in trip.handling_operations:
                errors.extend(self._check_negative_operation(trip, operation))
        if plan.reported_cost is not None:
            errors.extend(self._check_negative_reported_cost(plan.reported_cost))
        return errors

    def _check_negative_parts(
        self,
        parts: tuple[DemandPart, ...],
    ) -> list[ValidationIssue]:
        """Report negative demand-part desi values."""

        return [
            self._issue(
                "negative_desi",
                "Demand part desisi negatif olamaz.",
                demand_id=part.demand_id,
                related_values={"part_id": part.part_id, "desi": part.desi},
            )
            for part in parts
            if part.desi < 0
        ]

    def _check_negative_leg_values(
        self,
        trip: VehicleTrip,
        leg: VehicleLeg,
        parts: dict[str, DemandPart],
    ) -> list[ValidationIssue]:
        """Report negative travel duration and assignment desi on one leg."""

        errors: list[ValidationIssue] = []
        if leg.arrival_at < leg.departure_at:
            errors.append(
                self._issue(
                    "negative_travel_duration",
                    "Varış zamanı çıkış zamanından önce olamaz.",
                    vehicle_id=trip.vehicle_id,
                    event_date=leg.departure_at.date(),
                    trip_id=trip.trip_id,
                    leg_id=leg.leg_id,
                    related_values={
                        "departure_at": leg.departure_at.isoformat(),
                        "arrival_at": leg.arrival_at.isoformat(),
                    },
                )
            )
        for assignment in leg.assignments:
            if assignment.desi >= 0:
                continue
            part = parts.get(assignment.demand_part_id)
            errors.append(
                self._issue(
                    "negative_desi",
                    "Araç bacağı assignment desisi negatif olamaz.",
                    demand_id=part.demand_id if part else None,
                    vehicle_id=trip.vehicle_id,
                    event_date=leg.departure_at.date(),
                    trip_id=trip.trip_id,
                    leg_id=leg.leg_id,
                    related_values={
                        "part_id": assignment.demand_part_id,
                        "desi": assignment.desi,
                    },
                )
            )
        return errors

    def _check_negative_operation(
        self,
        trip: VehicleTrip,
        operation: HandlingOperation,
    ) -> list[ValidationIssue]:
        """Report negative desi or duration on one handling operation."""

        errors: list[ValidationIssue] = []
        context = {
            "vehicle_id": operation.vehicle_id,
            "event_date": operation.start_at.date(),
            "trip_id": trip.trip_id,
        }
        if operation.desi < 0:
            errors.append(
                self._issue(
                    "negative_desi",
                    "Elleçleme desisi negatif olamaz.",
                    **context,
                    related_values={
                        "operation_id": operation.operation_id,
                        "desi": operation.desi,
                    },
                )
            )
        if operation.end_at < operation.start_at:
            errors.append(
                self._issue(
                    "negative_handling_duration",
                    "Elleçleme bitişi başlangıçtan önce olamaz.",
                    **context,
                    related_values={
                        "operation_id": operation.operation_id,
                        "start_at": operation.start_at.isoformat(),
                        "end_at": operation.end_at.isoformat(),
                    },
                )
            )
        return errors

    def _check_negative_reported_cost(
        self,
        reported: CostBreakdown,
    ) -> list[ValidationIssue]:
        """Report negative totals and line-level reported cost components."""

        errors: list[ValidationIssue] = []
        totals = {
            "mandatory_rented_hourly_cost_tl": reported.mandatory_rented_hourly_cost_tl,
            "mandatory_rented_distance_cost_tl": reported.mandatory_rented_distance_cost_tl,
            "spot_hourly_cost_tl": reported.spot_hourly_cost_tl,
            "spot_distance_cost_tl": reported.spot_distance_cost_tl,
            "sla_penalty_tl": reported.sla_penalty_tl,
            "total_cost_tl": reported.total_cost_tl,
        }
        for field_name, value in totals.items():
            if value < 0:
                errors.append(
                    self._issue(
                        "negative_cost",
                        "Raporlanan maliyet negatif olamaz.",
                        related_values={"field": field_name, "value": value},
                    )
                )
        for line in reported.vehicle_cost_lines:
            if any(
                value < 0
                for value in (
                    line.usage_hours,
                    line.charged_hours,
                    line.distance_km,
                    line.hourly_cost_tl,
                    line.distance_cost_tl,
                    line.total_cost_tl,
                )
            ):
                errors.append(
                    self._issue(
                        "negative_cost",
                        "Araç maliyet satırında negatif süre, mesafe veya maliyet var.",
                        vehicle_id=line.vehicle_id,
                        related_values={"cost_unit_id": line.cost_unit_id},
                    )
                )
        for line in reported.sla_penalty_lines:
            if any(
                value < 0
                for value in (
                    line.delivered_desi,
                    line.delay_hours,
                    line.charged_delay_hours,
                    line.penalty_tl,
                )
            ):
                errors.append(
                    self._issue(
                        "negative_cost",
                        "SLA maliyet satırında negatif desi, süre veya ceza var.",
                        demand_id=line.demand_id,
                        event_date=line.delivered_at.date(),
                        related_values={
                            "part_id": line.demand_part_id,
                            "penalty_tl": line.penalty_tl,
                        },
                    )
                )
        return errors

    def _check_vehicle_identity(
        self,
        plan: TransportPlan,
    ) -> list[ValidationIssue]:
        """Ensure physical IDs are coherent, unique where required and usable."""

        errors = self._check_trip_structure(plan.trips)
        grouped: dict[str, list[VehicleTrip]] = defaultdict(list)
        for trip in plan.trips:
            grouped[trip.vehicle_id].append(trip)
        for vehicle_id, trips in grouped.items():
            errors.extend(self._check_one_vehicle_identity(vehicle_id, trips))
        return errors

    def _check_trip_structure(
        self,
        trips: tuple[VehicleTrip, ...],
    ) -> list[ValidationIssue]:
        """Report invalid ownership labels and trips with no physical movement."""

        errors: list[ValidationIssue] = []
        for trip in trips:
            if trip.ownership_type not in {"mandatory_rented", "spot"}:
                errors.append(
                    self._issue(
                        "invalid_vehicle_ownership",
                        "Araç sahiplik türü mandatory_rented veya spot olmalıdır.",
                        vehicle_id=trip.vehicle_id,
                        event_date=self._trip_start_date(trip),
                        trip_id=trip.trip_id,
                        related_values={"ownership_type": trip.ownership_type},
                    )
                )
            if not trip.legs:
                errors.append(
                    self._issue(
                        "empty_trip",
                        "VehicleTrip en az bir fiziksel araç bacağı içermelidir.",
                        vehicle_id=trip.vehicle_id,
                        event_date=self._trip_start_date(trip),
                        trip_id=trip.trip_id,
                        related_values={"leg_count": 0},
                    )
                )
        return errors

    def _check_one_vehicle_identity(
        self,
        vehicle_id: str,
        trips: list[VehicleTrip],
    ) -> list[ValidationIssue]:
        """Check identity, overlap and physical location across vehicle trips."""

        errors: list[ValidationIssue] = []
        identities = {(trip.vehicle_type, trip.ownership_type) for trip in trips}
        event_date = min(
            (self._trip_start_date(trip) for trip in trips if trip.legs),
            default=None,
        )
        if len(identities) > 1:
            errors.append(
                self._issue(
                    "inconsistent_vehicle_identity",
                    "Aynı fiziksel araç farklı tür veya sahiplikle kullanılmış.",
                    vehicle_id=vehicle_id,
                    event_date=event_date,
                    related_values={"identities": sorted(identities)},
                )
            )
        if any(trip.ownership_type == "mandatory_rented" for trip in trips) and len(trips) > 1:
            errors.append(
                self._issue(
                    "mandatory_vehicle_id_reused",
                    "Her günlük zorunlu kiralık ayrı fiziksel vehicle_id taşımalıdır.",
                    vehicle_id=vehicle_id,
                    event_date=event_date,
                    related_values={
                        "trip_ids": [trip.trip_id for trip in trips],
                        "departure_dates": sorted(
                            {
                                trip.legs[0].departure_at.date().isoformat()
                                for trip in trips
                                if trip.legs
                            }
                        ),
                    },
                )
            )
        errors.extend(self._check_vehicle_trip_overlap(vehicle_id, trips))
        errors.extend(self._check_vehicle_trip_location(vehicle_id, trips))
        return errors

    def _check_vehicle_trip_overlap(
        self,
        vehicle_id: str,
        trips: list[VehicleTrip],
    ) -> list[ValidationIssue]:
        """Reject simultaneous trip envelopes for the same physical vehicle."""

        usable = [trip for trip in trips if trip.legs]
        ordered = sorted(usable, key=lambda trip: self._trip_activity_bounds(trip)[0])
        errors: list[ValidationIssue] = []
        for previous, following in zip(ordered, ordered[1:], strict=False):
            previous_end = self._trip_activity_bounds(previous)[1]
            following_start = self._trip_activity_bounds(following)[0]
            if following_start < previous_end:
                errors.append(
                    self._issue(
                        "physical_vehicle_trip_overlap",
                        "Aynı fiziksel araç eşzamanlı iki tripte kullanılmış.",
                        vehicle_id=vehicle_id,
                        event_date=following_start.date(),
                        trip_id=following.trip_id,
                        related_values={
                            "previous_trip_id": previous.trip_id,
                            "previous_end_at": previous_end.isoformat(),
                            "following_start_at": following_start.isoformat(),
                        },
                    )
                )
        return errors

    def _check_vehicle_trip_location(
        self,
        vehicle_id: str,
        trips: list[VehicleTrip],
    ) -> list[ValidationIssue]:
        """Require a physical vehicle's next trip to start where it stopped.

        Trips are ordered by their first departure time.  An empty
        repositioning trip therefore repairs location continuity explicitly;
        idle time alone never moves a vehicle between transfer centres.
        """

        usable = sorted(
            (trip for trip in trips if trip.legs),
            key=lambda trip: trip.legs[0].departure_at,
        )
        errors: list[ValidationIssue] = []
        for previous, following in zip(usable, usable[1:], strict=False):
            previous_leg = previous.legs[-1]
            following_leg = following.legs[0]
            if previous_leg.destination != following_leg.origin:
                errors.append(
                    self._issue(
                        "physical_vehicle_location_discontinuity",
                        "Aynı fiziksel aracın sonraki tripi son konumundan başlamıyor.",
                        vehicle_id=vehicle_id,
                        event_date=following_leg.departure_at.date(),
                        trip_id=following.trip_id,
                        leg_id=following_leg.leg_id,
                        related_values={
                            "previous_trip_id": previous.trip_id,
                            "previous_destination": previous_leg.destination,
                            "following_origin": following_leg.origin,
                        },
                    )
                )
            errors.extend(
                self._check_cross_trip_loading_time(
                    previous,
                    previous_leg,
                    following,
                    following_leg,
                )
            )
        return errors

    def _check_cross_trip_loading_time(
        self,
        previous_trip: VehicleTrip,
        previous_leg: VehicleLeg,
        following_trip: VehicleTrip,
        following_leg: VehicleLeg,
    ) -> list[ValidationIssue]:
        """Reject a next-trip load that starts before the vehicle is available."""

        operations = {
            operation.operation_id: operation
            for trip in (previous_trip, following_trip)
            for operation in trip.handling_operations
        }
        unload = operations.get(previous_leg.unloading_operation_id or "")
        load = operations.get(following_leg.loading_operation_id or "")
        available_at = unload.end_at if unload else previous_leg.arrival_at
        if load is None or load.start_at >= available_at:
            return []
        return [
            self._issue(
                "handling_before_vehicle_arrival",
                "Sonraki trip yüklemesi fiziksel araç kullanılabilir olmadan başladı.",
                vehicle_id=following_trip.vehicle_id,
                event_date=load.start_at.date(),
                trip_id=following_trip.trip_id,
                leg_id=following_leg.leg_id,
                related_values={
                    "previous_trip_id": previous_trip.trip_id,
                    "vehicle_available_at": available_at.isoformat(),
                    "load_start_at": load.start_at.isoformat(),
                },
            )
        ]

    def _check_assignments_and_capacity(
        self,
        plan: TransportPlan,
        index: _PlanIndex,
    ) -> list[ValidationIssue]:
        """Validate assignment references and carried load on every leg."""

        errors: list[ValidationIssue] = []
        for trip in plan.trips:
            vehicle_type = self._vehicle_types.get(trip.vehicle_type)
            if vehicle_type is None:
                errors.append(
                    self._issue(
                        "unknown_vehicle_type",
                        "Plan araç türü statik araç ana verisinde bulunmuyor.",
                        vehicle_id=trip.vehicle_id,
                        event_date=self._trip_start_date(trip),
                        trip_id=trip.trip_id,
                        related_values={"vehicle_type": trip.vehicle_type},
                    )
                )
                continue

            for leg in trip.legs:
                errors.extend(
                    self._check_leg_capacity(trip, leg, vehicle_type.capacity_desi, index)
                )
                errors.extend(self._check_leg_assignment_refs(trip, leg, index))
        return errors

    def _check_leg_capacity(
        self,
        trip: VehicleTrip,
        leg: VehicleLeg,
        capacity_desi: float,
        index: _PlanIndex,
    ) -> list[ValidationIssue]:
        """Compare one leg's total carried desi with its vehicle capacity."""

        carried_desi = math.fsum(item.desi for item in leg.assignments)
        if carried_desi <= capacity_desi + self.policy.desi_tolerance:
            return []
        demand_ids = sorted(
            {
                index.parts[item.demand_part_id].demand_id
                for item in leg.assignments
                if item.demand_part_id in index.parts
            }
        )
        return [
            self._issue(
                "vehicle_capacity_exceeded",
                "Araç bacağındaki toplam desi araç kapasitesini aşıyor.",
                demand_id=demand_ids[0] if len(demand_ids) == 1 else None,
                vehicle_id=trip.vehicle_id,
                event_date=leg.departure_at.date(),
                trip_id=trip.trip_id,
                leg_id=leg.leg_id,
                related_values={
                    "carried_desi": carried_desi,
                    "capacity_desi": capacity_desi,
                    "excess_desi": carried_desi - capacity_desi,
                    "demand_ids": demand_ids,
                },
            )
        ]

    def _check_leg_assignment_refs(
        self,
        trip: VehicleTrip,
        leg: VehicleLeg,
        index: _PlanIndex,
    ) -> list[ValidationIssue]:
        """Check assignment references, duplicates and per-part desi limits."""

        errors: list[ValidationIssue] = []
        seen_parts: set[str] = set()
        for assignment in leg.assignments:
            part = index.parts.get(assignment.demand_part_id)
            if part is None:
                errors.append(
                    self._issue(
                        "unknown_demand_part",
                        "Assignment bilinmeyen demand part kullanıyor.",
                        vehicle_id=trip.vehicle_id,
                        event_date=leg.departure_at.date(),
                        trip_id=trip.trip_id,
                        leg_id=leg.leg_id,
                        related_values={
                            "part_id": assignment.demand_part_id,
                            "desi": assignment.desi,
                        },
                    )
                )
                continue
            if assignment.demand_part_id in seen_parts:
                errors.append(self._duplicate_assignment_issue(trip, leg, part))
            seen_parts.add(assignment.demand_part_id)
            if abs(assignment.desi - part.desi) > self.policy.desi_tolerance:
                errors.append(
                    self._assignment_desi_mismatch_issue(
                        trip,
                        leg,
                        part,
                        assignment.desi,
                    )
                )
        return errors

    def _duplicate_assignment_issue(
        self,
        trip: VehicleTrip,
        leg: VehicleLeg,
        part: DemandPart,
    ) -> ValidationIssue:
        """Build one duplicate-assignment issue with physical context."""

        return self._issue(
            "duplicate_assignment_on_leg",
            "Aynı demand part aynı araç bacağında tekrarlanmış.",
            demand_id=part.demand_id,
            vehicle_id=trip.vehicle_id,
            event_date=leg.departure_at.date(),
            trip_id=trip.trip_id,
            leg_id=leg.leg_id,
            related_values={"part_id": part.part_id},
        )

    def _assignment_desi_mismatch_issue(
        self,
        trip: VehicleTrip,
        leg: VehicleLeg,
        part: DemandPart,
        assignment_desi: float,
    ) -> ValidationIssue:
        """Build one issue when an assignment changes an atomic part's desi."""

        return self._issue(
            "assignment_desi_mismatch",
            "Assignment desisi atomik DemandPart desisiyle eşit olmalıdır.",
            demand_id=part.demand_id,
            vehicle_id=trip.vehicle_id,
            event_date=leg.departure_at.date(),
            trip_id=trip.trip_id,
            leg_id=leg.leg_id,
            related_values={
                "part_id": part.part_id,
                "assignment_desi": assignment_desi,
                "part_desi": part.desi,
                "difference_desi": assignment_desi - part.desi,
            },
        )

    def _check_time_sequence_and_matrix(
        self,
        plan: TransportPlan,
        index: _PlanIndex,
    ) -> list[ValidationIssue]:
        """Check load/depart/arrive/unload ordering and matrix travel times."""

        errors = self._check_handling_reference_usage(plan)
        for trip in plan.trips:
            # Declared leg order represents the physical sequence of one trip.
            for previous, following in zip(trip.legs, trip.legs[1:], strict=False):
                if previous.destination != following.origin:
                    errors.append(
                        self._issue(
                            "trip_center_discontinuity",
                            "Ardışık araç bacaklarının merkezleri birbirine bağlanmıyor.",
                            vehicle_id=trip.vehicle_id,
                            event_date=following.departure_at.date(),
                            trip_id=trip.trip_id,
                            leg_id=following.leg_id,
                            related_values={
                                "previous_destination": previous.destination,
                                "next_origin": following.origin,
                            },
                        )
                    )
                if following.departure_at < previous.arrival_at:
                    errors.append(
                        self._issue(
                            "trip_time_overlap",
                            "Aynı araç önceki bacağa varmadan sonraki bacağa çıkmış.",
                            vehicle_id=trip.vehicle_id,
                            event_date=following.departure_at.date(),
                            trip_id=trip.trip_id,
                            leg_id=following.leg_id,
                            related_values={
                                "previous_arrival": previous.arrival_at.isoformat(),
                                "next_departure": following.departure_at.isoformat(),
                            },
                        )
                    )
                errors.extend(
                    self._check_intermediate_handling_sequence(
                        trip,
                        previous,
                        following,
                        index,
                    )
                )

            for operation in trip.handling_operations:
                errors.extend(self._check_handling_operation(trip, operation))
            for leg in trip.legs:
                errors.extend(self._check_leg_matrix_time(trip, leg))
                errors.extend(self._check_leg_handling_links(trip, leg, index))
        return errors

    def _check_handling_reference_usage(
        self,
        plan: TransportPlan,
    ) -> list[ValidationIssue]:
        """Require each declared physical handling operation to be linked once."""

        references: Counter[str] = Counter()
        declared: dict[str, tuple[VehicleTrip, HandlingOperation]] = {}
        for trip in plan.trips:
            for operation in trip.handling_operations:
                declared.setdefault(operation.operation_id, (trip, operation))
            for leg in trip.legs:
                if leg.loading_operation_id is not None:
                    references[leg.loading_operation_id] += 1
                if leg.unloading_operation_id is not None:
                    references[leg.unloading_operation_id] += 1

        errors: list[ValidationIssue] = []
        for operation_id, count in references.items():
            if count <= 1 or operation_id not in declared:
                continue
            trip, operation = declared[operation_id]
            errors.append(
                self._issue(
                    "handling_operation_reused",
                    "Tek fiziksel elleçleme birden fazla bacak referansında kullanılmış.",
                    vehicle_id=operation.vehicle_id,
                    event_date=operation.start_at.date(),
                    trip_id=trip.trip_id,
                    related_values={"operation_id": operation_id, "count": count},
                )
            )
        for operation_id, (trip, operation) in declared.items():
            if references[operation_id] == 0:
                errors.append(
                    self._issue(
                        "orphan_handling_operation",
                        "Elleçleme operasyonu hiçbir araç bacağına bağlı değil.",
                        vehicle_id=operation.vehicle_id,
                        event_date=operation.start_at.date(),
                        trip_id=trip.trip_id,
                        related_values={"operation_id": operation_id},
                    )
                )
        return errors

    def _check_intermediate_handling_sequence(
        self,
        trip: VehicleTrip,
        incoming_leg: VehicleLeg,
        outgoing_leg: VehicleLeg,
        index: _PlanIndex,
    ) -> list[ValidationIssue]:
        """Require one vehicle's reload to wait for its unload to finish.

        The comparison is deliberately scoped to the same physical vehicle at
        one intermediate stop.  Equality is valid because the rules impose no
        minimum waiting time; operations belonging to different vehicles may
        still overlap freely.
        """

        unload = index.operations.get(incoming_leg.unloading_operation_id or "")
        load = index.operations.get(outgoing_leg.loading_operation_id or "")
        if load is None:
            return []
        available_at = unload.end_at if unload else incoming_leg.arrival_at
        if load.start_at >= available_at:
            return []
        if unload is None:
            return [
                self._issue(
                    "handling_before_vehicle_arrival",
                    "Ara merkez yüklemesi fiziksel araç merkeze varmadan başladı.",
                    vehicle_id=trip.vehicle_id,
                    event_date=load.start_at.date(),
                    trip_id=trip.trip_id,
                    leg_id=outgoing_leg.leg_id,
                    related_values={
                        "previous_arrival_at": incoming_leg.arrival_at.isoformat(),
                        "loading_operation_id": load.operation_id,
                        "load_start_at": load.start_at.isoformat(),
                    },
                )
            ]
        return [
            self._issue(
                "reload_started_before_unload_finished",
                "Ara merkez yüklemesi aynı aracın boşaltması bitmeden başlamış.",
                vehicle_id=trip.vehicle_id,
                event_date=load.start_at.date(),
                trip_id=trip.trip_id,
                leg_id=outgoing_leg.leg_id,
                related_values={
                    "unloading_operation_id": unload.operation_id,
                    "unload_end_at": unload.end_at.isoformat(),
                    "loading_operation_id": load.operation_id,
                    "load_start_at": load.start_at.isoformat(),
                },
            )
        ]

    def _check_cargo_flow(
        self,
        plan: TransportPlan,
        index: _PlanIndex,
    ) -> list[ValidationIssue]:
        """Validate every atomic demand part from first load to final unload.

        The check connects leg assignments into a physical chain.  It permits
        a zero-wait transfer after unloading and an unloaded-free continuation
        only on the immediately following leg of the same physical vehicle.
        """

        occurrences = self._collect_cargo_occurrences(plan, index)
        leg_positions = self._vehicle_leg_positions(plan)
        errors: list[ValidationIssue] = []
        for part in index.parts.values():
            part_occurrences = sorted(
                occurrences.get(part.part_id, ()),
                key=self._cargo_occurrence_sort_key,
            )
            errors.extend(
                self._check_one_part_flow(
                    part,
                    part_occurrences,
                    leg_positions,
                )
            )
        return errors

    def _collect_cargo_occurrences(
        self,
        plan: TransportPlan,
        index: _PlanIndex,
    ) -> dict[str, list[_CargoOccurrence]]:
        """Group assignment occurrences with their linked handling records."""

        grouped: dict[str, list[_CargoOccurrence]] = defaultdict(list)
        for trip in plan.trips:
            for leg in trip.legs:
                load = index.operations.get(leg.loading_operation_id or "")
                unload = index.operations.get(leg.unloading_operation_id or "")
                for assignment in leg.assignments:
                    if assignment.demand_part_id not in index.parts:
                        continue
                    grouped[assignment.demand_part_id].append(
                        _CargoOccurrence(trip, leg, assignment, load, unload)
                    )
        return grouped

    @staticmethod
    def _vehicle_leg_positions(
        plan: TransportPlan,
    ) -> dict[tuple[str, str, str], int]:
        """Index each leg's chronological position for one physical vehicle."""

        grouped: dict[str, list[tuple[VehicleTrip, VehicleLeg]]] = defaultdict(list)
        for trip in plan.trips:
            grouped[trip.vehicle_id].extend((trip, leg) for leg in trip.legs)
        positions: dict[tuple[str, str, str], int] = {}
        for vehicle_id, items in grouped.items():
            ordered = sorted(
                items,
                key=lambda item: (
                    item[1].departure_at,
                    item[1].arrival_at,
                    item[0].trip_id,
                    item[1].leg_id,
                ),
            )
            for position, (trip, leg) in enumerate(ordered):
                positions[(vehicle_id, trip.trip_id, leg.leg_id)] = position
        return positions

    def _check_one_part_flow(
        self,
        part: DemandPart,
        occurrences: list[_CargoOccurrence],
        leg_positions: dict[tuple[str, str, str], int],
    ) -> list[ValidationIssue]:
        """Check one atomic part's origin, transitions and unique final event."""

        if not occurrences:
            return [
                self._issue(
                    "cargo_flow_discontinuity",
                    "DemandPart hiçbir fiziksel araç bacağında taşınmamış.",
                    demand_id=part.demand_id,
                    related_values={"part_id": part.part_id},
                )
            ]
        errors = self._check_part_initial_load(part, occurrences[0])
        for previous, current in zip(occurrences, occurrences[1:], strict=False):
            errors.extend(
                self._check_part_transition(
                    part,
                    previous,
                    current,
                    leg_positions,
                )
            )
        errors.extend(self._check_part_final_state(part, occurrences))
        return errors

    def _check_part_initial_load(
        self,
        part: DemandPart,
        occurrence: _CargoOccurrence,
    ) -> list[ValidationIssue]:
        """Require the first occurrence to load at the true origin after ready."""

        demand = self._demands.get(part.demand_id)
        if demand is None:
            return []
        errors: list[ValidationIssue] = []
        if not occurrence.assignment.loaded_at_origin:
            errors.append(
                self._cargo_issue(
                    "cargo_flow_discontinuity",
                    "DemandPart ilk kez yüklenmeden taşınmış veya teslim edilmiş.",
                    part,
                    occurrence,
                )
            )
        if occurrence.leg.origin != demand.origin:
            errors.append(
                self._cargo_issue(
                    "cargo_teleportation",
                    "DemandPart ilk kez gerçek çıkış merkezi dışında ortaya çıkmış.",
                    part,
                    occurrence,
                    expected_center=demand.origin,
                    actual_center=occurrence.leg.origin,
                )
            )
        if occurrence.assignment.loaded_at_origin and occurrence.load_operation is None:
            errors.append(
                self._cargo_issue(
                    "cargo_flow_discontinuity",
                    "İlk yükleme bayrağı için fiziksel yükleme operasyonu bulunmuyor.",
                    part,
                    occurrence,
                )
            )
        errors.extend(self._check_initial_load_time(part, occurrence, demand))
        return errors

    def _check_initial_load_time(
        self,
        part: DemandPart,
        occurrence: _CargoOccurrence,
        demand: Any,
    ) -> list[ValidationIssue]:
        """Reject a physical first load that starts before demand readiness."""

        load = occurrence.load_operation
        if not occurrence.assignment.loaded_at_origin or load is None:
            return []
        ready_at = datetime.combine(demand.demand_date, demand.time_slot)
        if load.start_at >= ready_at:
            return []
        return [
            self._cargo_issue(
                "load_before_demand_ready",
                "Talep hazır olmadan ilk yükleme başlatılmış.",
                part,
                occurrence,
                load_start_at=load.start_at.isoformat(),
                demand_ready_at=ready_at.isoformat(),
            )
        ]

    def _check_part_transition(
        self,
        part: DemandPart,
        previous: _CargoOccurrence,
        current: _CargoOccurrence,
        leg_positions: dict[tuple[str, str, str], int],
    ) -> list[ValidationIssue]:
        """Validate one part transition through a transfer or vehicle continuation."""

        errors: list[ValidationIssue] = []
        if previous.assignment.is_final_delivery:
            errors.append(
                self._cargo_issue(
                    "cargo_flow_discontinuity",
                    "DemandPart nihai teslimattan sonra yeniden taşınmış.",
                    part,
                    current,
                    previous_leg_id=previous.leg.leg_id,
                )
            )
        if previous.leg.destination != current.leg.origin:
            errors.append(
                self._cargo_issue(
                    "cargo_teleportation",
                    "DemandPart ardışık bacaklar arasında başka merkeze ışınlanmış.",
                    part,
                    current,
                    previous_center=previous.leg.destination,
                    current_center=current.leg.origin,
                )
            )
        if previous.assignment.unloaded_at_destination:
            errors.extend(self._check_transfer_transition(part, previous, current))
        else:
            errors.extend(
                self._check_on_vehicle_transition(
                    part,
                    previous,
                    current,
                    leg_positions,
                )
            )
        return errors

    def _check_transfer_transition(
        self,
        part: DemandPart,
        previous: _CargoOccurrence,
        current: _CargoOccurrence,
    ) -> list[ValidationIssue]:
        """Require an unloaded part to be reloaded after unload completion."""

        if not current.assignment.loaded_at_origin:
            return [
                self._cargo_issue(
                    "cargo_flow_discontinuity",
                    "Boşaltılan DemandPart sonraki bacak için yeniden yüklenmemiş.",
                    part,
                    current,
                    previous_leg_id=previous.leg.leg_id,
                )
            ]
        unload = previous.unload_operation
        load = current.load_operation
        if unload is None or load is None:
            return [
                self._cargo_issue(
                    "cargo_flow_discontinuity",
                    "Araç transferinin yükleme veya boşaltma operasyonu eksik.",
                    part,
                    current,
                    previous_leg_id=previous.leg.leg_id,
                )
            ]
        if load.start_at >= unload.end_at:
            return []
        return [
            self._cargo_issue(
                "cargo_flow_discontinuity",
                "Transfer yüklemesi önceki boşaltma tamamlanmadan başladı.",
                part,
                current,
                unload_end_at=unload.end_at.isoformat(),
                load_start_at=load.start_at.isoformat(),
            )
        ]

    def _check_on_vehicle_transition(
        self,
        part: DemandPart,
        previous: _CargoOccurrence,
        current: _CargoOccurrence,
        leg_positions: dict[tuple[str, str, str], int],
    ) -> list[ValidationIssue]:
        """Require unloaded-free cargo to remain on the next leg of one vehicle."""

        errors: list[ValidationIssue] = []
        if current.assignment.loaded_at_origin:
            errors.append(
                self._cargo_issue(
                    "cargo_flow_discontinuity",
                    "DemandPart boşaltılmadan aynı veya başka araca yeniden yüklenmiş.",
                    part,
                    current,
                    previous_leg_id=previous.leg.leg_id,
                )
            )
        if previous.trip.vehicle_id != current.trip.vehicle_id:
            errors.append(
                self._cargo_issue(
                    "cargo_flow_discontinuity",
                    "Boşaltılmayan DemandPart fiziksel araç değiştirmiş.",
                    part,
                    current,
                    previous_vehicle_id=previous.trip.vehicle_id,
                )
            )
        if not self._are_adjacent_vehicle_legs(previous, current, leg_positions):
            errors.append(
                self._cargo_issue(
                    "cargo_flow_discontinuity",
                    "DemandPart aynı aracın hemen sonraki bacağında devam etmiyor.",
                    part,
                    current,
                    previous_leg_id=previous.leg.leg_id,
                )
            )
        if current.leg.departure_at < previous.leg.arrival_at:
            errors.append(
                self._cargo_issue(
                    "cargo_flow_discontinuity",
                    "DemandPart önceki araç bacağı varmadan sonraki bacakta ilerlemiş.",
                    part,
                    current,
                    previous_arrival_at=previous.leg.arrival_at.isoformat(),
                )
            )
        return errors

    @staticmethod
    def _are_adjacent_vehicle_legs(
        previous: _CargoOccurrence,
        current: _CargoOccurrence,
        positions: dict[tuple[str, str, str], int],
    ) -> bool:
        """Return whether two occurrences use consecutive legs of one vehicle."""

        if previous.trip.vehicle_id != current.trip.vehicle_id:
            return False
        previous_key = (
            previous.trip.vehicle_id,
            previous.trip.trip_id,
            previous.leg.leg_id,
        )
        current_key = (
            current.trip.vehicle_id,
            current.trip.trip_id,
            current.leg.leg_id,
        )
        return positions.get(current_key) == positions.get(previous_key, -2) + 1

    def _check_part_final_state(
        self,
        part: DemandPart,
        occurrences: list[_CargoOccurrence],
    ) -> list[ValidationIssue]:
        """Require exactly one final unload at the original destination."""

        finals = [item for item in occurrences if item.assignment.is_final_delivery]
        if len(finals) != 1:
            return [
                self._cargo_issue(
                    "cargo_flow_discontinuity",
                    "DemandPart tam olarak bir nihai teslimatla bitmelidir.",
                    part,
                    occurrences[-1],
                    final_delivery_count=len(finals),
                )
            ]
        final = finals[0]
        errors: list[ValidationIssue] = []
        if final is not occurrences[-1]:
            errors.append(
                self._cargo_issue(
                    "cargo_flow_discontinuity",
                    "Nihai teslimat DemandPart hareket zincirinin son olayı değil.",
                    part,
                    final,
                    last_leg_id=occurrences[-1].leg.leg_id,
                )
            )
        demand = self._demands.get(part.demand_id)
        if demand is not None and final.leg.destination != demand.destination:
            errors.append(
                self._cargo_issue(
                    "cargo_teleportation",
                    "DemandPart gerçek varış merkezi dışında nihai teslim edilmiş.",
                    part,
                    final,
                    expected_center=demand.destination,
                    actual_center=final.leg.destination,
                )
            )
        if not final.assignment.unloaded_at_destination:
            errors.append(
                self._cargo_issue(
                    "cargo_flow_discontinuity",
                    "Nihai teslimat fiziksel boşaltma olmadan işaretlenmiş.",
                    part,
                    final,
                )
            )
        return errors

    @staticmethod
    def _cargo_occurrence_sort_key(
        occurrence: _CargoOccurrence,
    ) -> tuple[datetime, datetime, str, str]:
        """Return deterministic chronological ordering for cargo occurrences."""

        return (
            occurrence.leg.departure_at,
            occurrence.leg.arrival_at,
            occurrence.trip.trip_id,
            occurrence.leg.leg_id,
        )

    def _cargo_issue(
        self,
        code: str,
        message: str,
        part: DemandPart,
        occurrence: _CargoOccurrence,
        **related_values: Any,
    ) -> ValidationIssue:
        """Build a cargo-flow issue with demand, vehicle, date and part context."""

        return self._issue(
            code,
            message,
            demand_id=part.demand_id,
            vehicle_id=occurrence.trip.vehicle_id,
            event_date=occurrence.leg.departure_at.date(),
            trip_id=occurrence.trip.trip_id,
            leg_id=occurrence.leg.leg_id,
            related_values={"part_id": part.part_id, **related_values},
        )

    def _check_handling_operation(
        self,
        trip: VehicleTrip,
        operation: HandlingOperation,
    ) -> list[ValidationIssue]:
        """Validate operation ownership, type and desi-based duration."""

        errors: list[ValidationIssue] = []
        if operation.vehicle_id != trip.vehicle_id:
            errors.append(
                self._issue(
                    "handling_vehicle_mismatch",
                    "Elleçleme işlemi bağlı olduğu trip ile aynı araca ait değil.",
                    vehicle_id=operation.vehicle_id,
                    event_date=operation.start_at.date(),
                    trip_id=trip.trip_id,
                    related_values={
                        "operation_id": operation.operation_id,
                        "trip_vehicle_id": trip.vehicle_id,
                    },
                )
            )
        if operation.operation_type not in {"load", "unload"}:
            errors.append(
                self._issue(
                    "invalid_handling_type",
                    "Elleçleme türü load veya unload olmalıdır.",
                    vehicle_id=operation.vehicle_id,
                    event_date=operation.start_at.date(),
                    trip_id=trip.trip_id,
                    related_values={
                        "operation_id": operation.operation_id,
                        "operation_type": operation.operation_type,
                    },
                )
            )
        if len(operation.demand_part_ids) != len(set(operation.demand_part_ids)):
            errors.append(
                self._issue(
                    "duplicate_handling_part_id",
                    "Aynı demand part tek elleçleme grubunda tekrarlanmış.",
                    vehicle_id=operation.vehicle_id,
                    event_date=operation.start_at.date(),
                    trip_id=trip.trip_id,
                    related_values={
                        "operation_id": operation.operation_id,
                        "demand_part_ids": operation.demand_part_ids,
                    },
                )
            )

        actual_seconds = (operation.end_at - operation.start_at).total_seconds()
        expected_seconds = (
            operation.desi
            * self.policy.handling_minutes_per_desi
            * SECONDS_PER_MINUTE
        )
        if (
            actual_seconds >= 0
            and abs(actual_seconds - expected_seconds)
            > self.policy.handling_duration_tolerance_seconds
        ):
            errors.append(
                self._issue(
                    "handling_duration_mismatch",
                    "Elleçleme süresi desi × dakika/desi formülüyle uyuşmuyor.",
                    vehicle_id=operation.vehicle_id,
                    event_date=operation.start_at.date(),
                    trip_id=trip.trip_id,
                    related_values={
                        "operation_id": operation.operation_id,
                        "desi": operation.desi,
                        "actual_seconds": actual_seconds,
                        "expected_seconds": expected_seconds,
                    },
                )
            )
        return errors

    def _check_leg_matrix_time(
        self,
        trip: VehicleTrip,
        leg: VehicleLeg,
    ) -> list[ValidationIssue]:
        """Compare one leg's travel duration with the directed matrix."""

        arc = self._arcs.get((leg.origin, leg.destination))
        if arc is None:
            return [
                self._issue(
                    "missing_travel_arc",
                    "Araç bacağı için mesafe/seyahat matrisi kaydı yok.",
                    vehicle_id=trip.vehicle_id,
                    event_date=leg.departure_at.date(),
                    trip_id=trip.trip_id,
                    leg_id=leg.leg_id,
                    related_values={
                        "origin": leg.origin,
                        "destination": leg.destination,
                    },
                )
            ]
        try:
            expected_hours = arc.travel_hours_for(trip.vehicle_type)
        except KeyError:
            return [
                self._issue(
                    "missing_vehicle_travel_time",
                    "Araç türü için matris seyahat süresi bulunmuyor.",
                    vehicle_id=trip.vehicle_id,
                    event_date=leg.departure_at.date(),
                    trip_id=trip.trip_id,
                    leg_id=leg.leg_id,
                    related_values={"vehicle_type": trip.vehicle_type},
                )
            ]

        actual_hours = (
            leg.arrival_at - leg.departure_at
        ).total_seconds() / SECONDS_PER_HOUR
        tolerance_hours = (
            self.policy.travel_time_tolerance_minutes / MINUTES_PER_HOUR
        )
        if actual_hours >= 0 and abs(actual_hours - expected_hours) > tolerance_hours:
            return [
                self._issue(
                    "travel_time_mismatch",
                    "Araç bacağı seyahat süresi matrisle uyuşmuyor.",
                    vehicle_id=trip.vehicle_id,
                    event_date=leg.departure_at.date(),
                    trip_id=trip.trip_id,
                    leg_id=leg.leg_id,
                    related_values={
                        "actual_hours": actual_hours,
                        "expected_hours": expected_hours,
                        "tolerance_minutes": self.policy.travel_time_tolerance_minutes,
                    },
                )
            ]
        return []

    def _check_leg_handling_links(
        self,
        trip: VehicleTrip,
        leg: VehicleLeg,
        index: _PlanIndex,
    ) -> list[ValidationIssue]:
        """Check linked load/unload operations, amounts and chronological order."""

        errors: list[ValidationIssue] = []
        loaded_assignments = [
            assignment
            for assignment in leg.assignments
            if assignment.loaded_at_origin
        ]
        unloaded_assignments = [
            assignment
            for assignment in leg.assignments
            if assignment.unloaded_at_destination
        ]
        errors.extend(
            self._check_one_handling_link(
                trip=trip,
                leg=leg,
                operation_id=leg.loading_operation_id,
                expected_type="load",
                expected_center=leg.origin,
                expected_desi=math.fsum(item.desi for item in loaded_assignments),
                expected_part_ids={item.demand_part_id for item in loaded_assignments},
                index=index,
            )
        )
        errors.extend(
            self._check_one_handling_link(
                trip=trip,
                leg=leg,
                operation_id=leg.unloading_operation_id,
                expected_type="unload",
                expected_center=leg.destination,
                expected_desi=math.fsum(item.desi for item in unloaded_assignments),
                expected_part_ids={item.demand_part_id for item in unloaded_assignments},
                index=index,
            )
        )
        return errors

    def _check_one_handling_link(
        self,
        *,
        trip: VehicleTrip,
        leg: VehicleLeg,
        operation_id: str | None,
        expected_type: str,
        expected_center: str,
        expected_desi: float,
        expected_part_ids: set[str],
        index: _PlanIndex,
    ) -> list[ValidationIssue]:
        """Validate one load/unload reference used by a vehicle leg."""

        errors: list[ValidationIssue] = []
        if operation_id is None:
            if expected_desi > self.policy.desi_tolerance:
                errors.append(
                    self._issue(
                        f"missing_{expected_type}_operation",
                        "Pozitif desi için bağlı elleçleme işlemi bulunmuyor.",
                        vehicle_id=trip.vehicle_id,
                        event_date=leg.departure_at.date(),
                        trip_id=trip.trip_id,
                        leg_id=leg.leg_id,
                        related_values={
                            "expected_type": expected_type,
                            "expected_desi": expected_desi,
                        },
                    )
                )
            return errors

        operation = index.operations.get(operation_id)
        if operation is None:
            return [
                self._issue(
                    "unknown_handling_operation",
                    "Araç bacağı bilinmeyen elleçleme kimliğine başvuruyor.",
                    vehicle_id=trip.vehicle_id,
                    event_date=leg.departure_at.date(),
                    trip_id=trip.trip_id,
                    leg_id=leg.leg_id,
                    related_values={"operation_id": operation_id},
                )
            ]

        errors.extend(
            self._check_handling_link_metadata(
                trip,
                leg,
                operation,
                expected_type,
                expected_center,
                index,
            )
        )
        errors.extend(
            self._check_handling_link_quantity(
                trip,
                leg,
                operation,
                expected_desi,
                expected_part_ids,
            )
        )
        errors.extend(
            self._check_handling_link_sequence(
                trip,
                leg,
                operation,
                expected_type,
            )
        )
        return errors

    def _check_handling_link_metadata(
        self,
        trip: VehicleTrip,
        leg: VehicleLeg,
        operation: HandlingOperation,
        expected_type: str,
        expected_center: str,
        index: _PlanIndex,
    ) -> list[ValidationIssue]:
        """Check one linked operation's trip, type and center identity."""

        errors: list[ValidationIssue] = []
        if index.trip_by_operation.get(operation.operation_id) is not trip:
            errors.append(
                self._issue(
                    "handling_trip_mismatch",
                    "Araç bacağı başka bir tripteki elleçlemeye bağlanmış.",
                    vehicle_id=trip.vehicle_id,
                    event_date=leg.departure_at.date(),
                    trip_id=trip.trip_id,
                    leg_id=leg.leg_id,
                    related_values={"operation_id": operation.operation_id},
                )
            )
        if operation.operation_type != expected_type:
            errors.append(
                self._issue(
                    "handling_type_mismatch",
                    "Bağlı elleçleme load/unload türüyle uyuşmuyor.",
                    vehicle_id=trip.vehicle_id,
                    event_date=operation.start_at.date(),
                    trip_id=trip.trip_id,
                    leg_id=leg.leg_id,
                    related_values={
                        "operation_id": operation.operation_id,
                        "actual_type": operation.operation_type,
                        "expected_type": expected_type,
                    },
                )
            )
        if operation.center != expected_center:
            errors.append(
                self._issue(
                    "handling_center_mismatch",
                    "Bağlı elleçleme araç bacağı merkeziyle uyuşmuyor.",
                    vehicle_id=trip.vehicle_id,
                    event_date=operation.start_at.date(),
                    trip_id=trip.trip_id,
                    leg_id=leg.leg_id,
                    related_values={
                        "operation_id": operation.operation_id,
                        "actual_center": operation.center,
                        "expected_center": expected_center,
                    },
                )
            )
        return errors

    def _check_handling_link_quantity(
        self,
        trip: VehicleTrip,
        leg: VehicleLeg,
        operation: HandlingOperation,
        expected_desi: float,
        expected_part_ids: set[str],
    ) -> list[ValidationIssue]:
        """Check grouped operation desi and demand-part membership."""

        errors: list[ValidationIssue] = []
        if abs(operation.desi - expected_desi) > self.policy.desi_tolerance:
            errors.append(
                self._issue(
                    "handling_desi_mismatch",
                    "Elleçleme desisi ilgili assignment toplamıyla uyuşmuyor.",
                    vehicle_id=trip.vehicle_id,
                    event_date=operation.start_at.date(),
                    trip_id=trip.trip_id,
                    leg_id=leg.leg_id,
                    related_values={
                        "operation_id": operation.operation_id,
                        "actual_desi": operation.desi,
                        "expected_desi": expected_desi,
                    },
                )
            )
        if set(operation.demand_part_ids) != expected_part_ids:
            errors.append(
                self._issue(
                    "handling_parts_mismatch",
                    "Elleçlemenin demand part listesi assignmentlarla uyuşmüyor.",
                    vehicle_id=trip.vehicle_id,
                    event_date=operation.start_at.date(),
                    trip_id=trip.trip_id,
                    leg_id=leg.leg_id,
                    related_values={
                        "operation_id": operation.operation_id,
                        "actual_part_ids": sorted(operation.demand_part_ids),
                        "expected_part_ids": sorted(expected_part_ids),
                    },
                )
            )
        return errors

    def _check_handling_link_sequence(
        self,
        trip: VehicleTrip,
        leg: VehicleLeg,
        operation: HandlingOperation,
        expected_type: str,
    ) -> list[ValidationIssue]:
        """Check load/departure or arrival/unload chronology for one leg."""

        if expected_type == "load" and operation.end_at > leg.departure_at:
            return [
                self._issue(
                    "invalid_load_departure_order",
                    "Yükleme tamamlanmadan araç çıkış yapmış.",
                    vehicle_id=trip.vehicle_id,
                    event_date=leg.departure_at.date(),
                    trip_id=trip.trip_id,
                    leg_id=leg.leg_id,
                    related_values={
                        "load_end": operation.end_at.isoformat(),
                        "departure_at": leg.departure_at.isoformat(),
                    },
                )
            ]
        if expected_type == "unload" and operation.start_at < leg.arrival_at:
            return [
                self._issue(
                    "invalid_arrival_unload_order",
                    "Araç varmadan boşaltma başlamış.",
                    vehicle_id=trip.vehicle_id,
                    event_date=leg.arrival_at.date(),
                    trip_id=trip.trip_id,
                    leg_id=leg.leg_id,
                    related_values={
                        "arrival_at": leg.arrival_at.isoformat(),
                        "unload_start": operation.start_at.isoformat(),
                    },
                )
            ]
        return []

    def _check_demand_delivery(
        self,
        plan: TransportPlan,
        index: _PlanIndex,
    ) -> tuple[list[ValidationIssue], tuple[_FinalDelivery, ...], dict[str, Any]]:
        """Check demand partitioning, final delivery, shortages and excess."""

        errors = self._check_demand_partitions(plan.demand_parts)
        deliveries = self._collect_final_deliveries(plan, index, errors)
        delivered_by_part: dict[str, float] = defaultdict(float)
        delivered_by_demand: dict[str, float] = defaultdict(float)
        for delivery in deliveries:
            delivered_by_part[delivery.part_id] += delivery.desi
            delivered_by_demand[delivery.demand_id] += delivery.desi
        errors.extend(self._check_part_delivery_totals(plan, delivered_by_part))
        demand_errors = self._check_demand_delivery_totals(delivered_by_demand)
        errors.extend(demand_errors)
        fully_delivered = not demand_errors
        metrics = {
            "required_desi": math.fsum(demand.desi for demand in self.problem.demands),
            "delivered_desi": math.fsum(delivered_by_demand.values()),
            "all_demands_fully_delivered": fully_delivered,
            "final_delivery_count": len(deliveries),
        }
        return errors, tuple(deliveries), metrics

    def _check_demand_partitions(
        self,
        parts: tuple[DemandPart, ...],
    ) -> list[ValidationIssue]:
        """Check that part desi totals equal each original demand desi."""

        errors: list[ValidationIssue] = []
        totals: dict[str, float] = defaultdict(float)
        for part in parts:
            if part.demand_id not in self._demands:
                errors.append(
                    self._issue(
                        "unknown_demand_id",
                        "Demand part bilinmeyen ana talebe bağlı.",
                        demand_id=part.demand_id,
                        related_values={"part_id": part.part_id, "desi": part.desi},
                    )
                )
            totals[part.demand_id] += part.desi
        for demand in self.problem.demands:
            difference = totals.get(demand.demand_id, 0.0) - demand.desi
            if abs(difference) <= self.policy.desi_tolerance:
                continue
            errors.append(
                self._issue(
                    "demand_parts_exceed_demand" if difference > 0 else "demand_parts_missing",
                    "Demand part toplamı ana talep desisiyle uyuşmuyor.",
                    demand_id=demand.demand_id,
                    event_date=demand.demand_date,
                    related_values={
                        "demand_desi": demand.desi,
                        "parts_desi": totals.get(demand.demand_id, 0.0),
                        "difference_desi": difference,
                    },
                )
            )
        return errors

    def _check_part_delivery_totals(
        self,
        plan: TransportPlan,
        delivered_by_part: dict[str, float],
    ) -> list[ValidationIssue]:
        """Check that each demand part is delivered exactly once in total desi."""

        errors: list[ValidationIssue] = []
        for part in plan.demand_parts:
            delivered = delivered_by_part.get(part.part_id, 0.0)
            difference = delivered - part.desi
            if abs(difference) <= self.policy.desi_tolerance:
                continue
            demand = self._demands.get(part.demand_id)
            errors.append(
                self._issue(
                    "demand_part_overdelivered" if difference > 0 else "demand_part_underdelivered",
                    "Demand part nihai teslim toplamı parça desisiyle uyuşmuyor.",
                    demand_id=part.demand_id,
                    event_date=demand.demand_date if demand else None,
                    related_values={
                        "part_id": part.part_id,
                        "part_desi": part.desi,
                        "delivered_desi": delivered,
                        "difference_desi": difference,
                    },
                )
            )
        return errors

    def _check_demand_delivery_totals(
        self,
        delivered_by_demand: dict[str, float],
    ) -> list[ValidationIssue]:
        """Check final delivered desi against every original demand."""

        errors: list[ValidationIssue] = []
        for demand in self.problem.demands:
            delivered = delivered_by_demand.get(demand.demand_id, 0.0)
            difference = delivered - demand.desi
            if abs(difference) <= self.policy.desi_tolerance:
                continue
            errors.append(
                self._issue(
                    "demand_overdelivered" if difference > 0 else "demand_underdelivered",
                    "Ana talebin nihai teslim toplamı gerekli desiyle uyuşmuyor.",
                    demand_id=demand.demand_id,
                    event_date=demand.demand_date,
                    related_values={
                        "required_desi": demand.desi,
                        "delivered_desi": delivered,
                        "difference_desi": difference,
                    },
                )
            )
        return errors

    def _collect_final_deliveries(
        self,
        plan: TransportPlan,
        index: _PlanIndex,
        errors: list[ValidationIssue],
    ) -> list[_FinalDelivery]:
        """Normalize valid final assignments and report invalid destinations."""

        deliveries: list[_FinalDelivery] = []
        for trip in plan.trips:
            for leg in trip.legs:
                unload = index.operations.get(leg.unloading_operation_id or "")
                delivered_at = unload.end_at if unload is not None else leg.arrival_at
                for assignment in leg.assignments:
                    if not assignment.is_final_delivery:
                        continue
                    part = index.parts.get(assignment.demand_part_id)
                    if part is None:
                        continue
                    demand = self._demands.get(part.demand_id)
                    if demand is None:
                        continue
                    if not assignment.unloaded_at_destination:
                        errors.append(
                            self._issue(
                                "final_delivery_not_unloaded",
                                "Nihai teslim olarak işaretlenen yük boşaltılmamış.",
                                demand_id=demand.demand_id,
                                vehicle_id=trip.vehicle_id,
                                event_date=delivered_at.date(),
                                trip_id=trip.trip_id,
                                leg_id=leg.leg_id,
                                related_values={"part_id": part.part_id},
                            )
                        )
                        continue
                    if leg.destination != demand.destination:
                        errors.append(
                            self._issue(
                                "wrong_final_destination",
                                "Nihai teslim orijinal talep varış merkezinde yapılmamış.",
                                demand_id=demand.demand_id,
                                vehicle_id=trip.vehicle_id,
                                event_date=delivered_at.date(),
                                trip_id=trip.trip_id,
                                leg_id=leg.leg_id,
                                related_values={
                                    "part_id": part.part_id,
                                    "actual_destination": leg.destination,
                                    "expected_destination": demand.destination,
                                },
                            )
                        )
                        continue
                    deliveries.append(
                        _FinalDelivery(
                            demand_id=demand.demand_id,
                            part_id=part.part_id,
                            vehicle_id=trip.vehicle_id,
                            trip_id=trip.trip_id,
                            leg_id=leg.leg_id,
                            destination=leg.destination,
                            desi=assignment.desi,
                            delivered_at=delivered_at,
                        )
                    )
        return deliveries

    def _check_handling_capacity(
        self,
        plan: TransportPlan,
    ) -> tuple[list[ValidationIssue], dict[str, Any]]:
        """Allocate handling across calendar days and check daily desi limits."""

        usage: dict[tuple[str, date], float] = defaultdict(float)
        vehicles: dict[tuple[str, date], set[str]] = defaultdict(set)
        errors: list[ValidationIssue] = []
        for trip in plan.trips:
            for operation in trip.handling_operations:
                if operation.center not in self._handling_capacities:
                    errors.append(
                        self._issue(
                            "unknown_handling_center",
                            "Elleçleme merkezi statik kapasite verisinde bulunmuyor.",
                            vehicle_id=operation.vehicle_id,
                            event_date=operation.start_at.date(),
                            trip_id=trip.trip_id,
                            related_values={
                                "operation_id": operation.operation_id,
                                "center": operation.center,
                            },
                        )
                    )
                    continue
                for operation_date, allocated_desi in self._allocate_handling_by_day(
                    operation
                ).items():
                    key = (operation.center, operation_date)
                    usage[key] += allocated_desi
                    vehicles[key].add(operation.vehicle_id)

        peak_ratio = 0.0
        for (center, operation_date), used_desi in sorted(usage.items()):
            capacity = self._handling_capacities[center]
            if capacity > 0:
                peak_ratio = max(peak_ratio, used_desi / capacity)
            elif used_desi > self.policy.desi_tolerance:
                peak_ratio = math.inf
            if used_desi > capacity + self.policy.desi_tolerance:
                errors.append(
                    self._issue(
                        "daily_handling_capacity_exceeded",
                        "TM günlük elleçleme kapasitesi aşılmış.",
                        event_date=operation_date,
                        related_values={
                            "center": center,
                            "used_desi": used_desi,
                            "capacity_desi": capacity,
                            "excess_desi": used_desi - capacity,
                            "vehicle_ids": sorted(vehicles[(center, operation_date)]),
                        },
                    )
                )
        return errors, {
            "handling_center_day_count": len(usage),
            "peak_handling_capacity_ratio": peak_ratio,
            "handling_usage_desi_by_center_day": {
                f"{center}|{operation_date.isoformat()}": used_desi
                for (center, operation_date), used_desi in sorted(usage.items())
            },
        }

    def _allocate_handling_by_day(
        self,
        operation: HandlingOperation,
    ) -> dict[date, float]:
        """Split operation desi proportionally when it crosses midnight."""

        duration_seconds = (operation.end_at - operation.start_at).total_seconds()
        if duration_seconds <= 0:
            return {operation.start_at.date(): operation.desi}

        allocations: dict[date, float] = defaultdict(float)
        cursor = operation.start_at
        while cursor < operation.end_at:
            next_midnight = datetime.combine(
                cursor.date() + timedelta(days=1),
                time.min,
            )
            segment_end = min(next_midnight, operation.end_at)
            segment_seconds = (segment_end - cursor).total_seconds()
            allocations[cursor.date()] += (
                operation.desi * segment_seconds / duration_seconds
            )
            cursor = segment_end
        return dict(allocations)

    def _check_daily_truck_quota(
        self,
        plan: TransportPlan,
    ) -> tuple[list[ValidationIssue], dict[str, Any]]:
        """Count Tır departure/arrival visits and check daily centre quotas."""

        usage: Counter[tuple[str, date]] = Counter()
        vehicles: dict[tuple[str, date], set[str]] = defaultdict(set)
        truck_legs_by_vehicle: dict[str, list[VehicleLeg]] = defaultdict(list)
        for trip in plan.trips:
            if trip.vehicle_type != self.policy.truck_vehicle_type or not trip.legs:
                continue
            truck_legs_by_vehicle[trip.vehicle_id].extend(trip.legs)
        for vehicle_id, legs in truck_legs_by_vehicle.items():
            ordered_legs = tuple(sorted(legs, key=lambda leg: leg.departure_at))
            for center, event_date in self._truck_visit_events(ordered_legs):
                usage[(center, event_date)] += 1
                vehicles[(center, event_date)].add(vehicle_id)

        errors: list[ValidationIssue] = []
        peak_ratio = 0.0
        for (center, event_date), visit_count in sorted(usage.items()):
            quota = self._truck_capacities.get(center)
            if quota is None:
                errors.append(
                    self._issue(
                        "unknown_truck_capacity_center",
                        "Tır ziyaret merkezi günlük kota verisinde bulunmuyor.",
                        event_date=event_date,
                        related_values={
                            "center": center,
                            "visit_count": visit_count,
                            "vehicle_ids": sorted(vehicles[(center, event_date)]),
                        },
                    )
                )
                continue
            if quota > 0:
                peak_ratio = max(peak_ratio, visit_count / quota)
            elif visit_count > 0:
                peak_ratio = math.inf
            if visit_count > quota:
                errors.append(
                    self._issue(
                        "daily_truck_quota_exceeded",
                        "TM günlük toplam Tır işlem/ziyaret kotası aşılmış.",
                        event_date=event_date,
                        related_values={
                            "center": center,
                            "visit_count": visit_count,
                            "quota": quota,
                            "excess_visits": visit_count - quota,
                            "vehicle_ids": sorted(vehicles[(center, event_date)]),
                        },
                    )
                )
        return errors, {
            "truck_center_day_count": len(usage),
            "peak_truck_quota_ratio": peak_ratio,
            "truck_visits_by_center_day": {
                f"{center}|{event_date.isoformat()}": visit_count
                for (center, event_date), visit_count in sorted(usage.items())
            },
        }

    def _truck_visit_events(
        self,
        legs: tuple[VehicleLeg, ...],
    ) -> list[tuple[str, date]]:
        """Count visits across all ordered legs of one physical Tır.

        Trip boundaries do not create a second visit when the same physical
        Tır remains at the center.  A movement away and later return does.
        Midnight-spanning handling never receives a fractional quota; the
        unresolved day choice for one continuous visit remains config-backed.
        """

        if not legs:
            return []
        events: list[tuple[str, date]] = [
            (legs[0].origin, legs[0].departure_at.date())
        ]
        for index, leg in enumerate(legs):
            following = legs[index + 1] if index + 1 < len(legs) else None
            continuous = (
                following is not None
                and leg.destination == following.origin
                and leg.arrival_at <= following.departure_at
            )
            if not continuous:
                events.append((leg.destination, leg.arrival_at.date()))
                if following is not None:
                    events.append((following.origin, following.departure_at.date()))
                continue

            arrival_date = leg.arrival_at.date()
            departure_date = following.departure_at.date()
            if arrival_date == departure_date:
                events.append((leg.destination, arrival_date))
                continue

            # Q-034 is unresolved; config makes the working interpretation
            # explicit instead of silently choosing a calendar day.
            if self.policy.continuous_visit_midnight_policy == "arrival_day":
                events.append((leg.destination, arrival_date))
            elif self.policy.continuous_visit_midnight_policy == "departure_day":
                events.append((following.origin, departure_date))
            else:
                events.append((leg.destination, arrival_date))
                events.append((following.origin, departure_date))
        return events

    def _check_mandatory_schedule(
        self,
        plan: TransportPlan,
        deliveries: tuple[_FinalDelivery, ...],
        all_demands_delivered: bool,
    ) -> tuple[list[ValidationIssue], dict[str, Any]]:
        """Check exact daily mandatory routes through the delivery horizon."""

        required_routes = self.problem.static.mandatory_rental_routes
        if not required_routes:
            unexpected = [
                self._issue(
                    "mandatory_route_mismatch",
                    "Statik zorunlu rota yokken mandatory_rented trip kullanılmış.",
                    vehicle_id=trip.vehicle_id,
                    event_date=self._trip_start_date(trip),
                    trip_id=trip.trip_id,
                )
                for trip in plan.trips
                if trip.ownership_type == "mandatory_rented"
            ]
            return unexpected, {
                "mandatory_missing_count": 0,
                "mandatory_excess_count": 0,
                "mandatory_post_window_missing_count": 0,
            }

        completion_at = (
            max((delivery.delivered_at for delivery in deliveries), default=None)
            if all_demands_delivered
            else None
        )
        end_date = self._mandatory_schedule_end_date(plan, completion_at)
        errors, actual, actual_vehicle_ids = self._collect_mandatory_actual(
            plan,
            required_routes,
            completion_at,
        )
        count_errors, count_metrics = self._check_mandatory_daily_counts(
            required_routes,
            actual,
            actual_vehicle_ids,
            end_date,
        )
        errors.extend(count_errors)
        return errors, count_metrics

    def _mandatory_schedule_end_date(
        self,
        plan: TransportPlan,
        completion_at: datetime | None,
    ) -> date:
        """Derive the inclusive daily mandatory schedule horizon."""

        latest_activity = self._latest_plan_activity(plan)
        candidate = completion_at or latest_activity
        end_date = max(
            self.policy.mandatory_forecast_end_date,
            candidate.date() if candidate else self.policy.mandatory_forecast_end_date,
        )
        if not self.policy.mandatory_completion_date_is_included:
            end_date -= timedelta(days=1)
        return end_date

    def _collect_mandatory_actual(
        self,
        plan: TransportPlan,
        required_routes: tuple[MandatoryRentalRoute, ...],
        completion_at: datetime | None,
    ) -> tuple[
        list[ValidationIssue],
        Counter[tuple[date, str, str, str]],
        dict[tuple[date, str, str, str], list[str]],
    ]:
        """Validate and count actual one-leg mandatory trips by route/day."""

        errors: list[ValidationIssue] = []
        actual: Counter[tuple[date, str, str, str]] = Counter()
        actual_vehicle_ids: dict[
            tuple[date, str, str, str],
            list[str],
        ] = defaultdict(list)
        allowed_route_keys = {
            (route.origin, route.destination, route.vehicle_type)
            for route in required_routes
        }
        for trip in plan.trips:
            if trip.ownership_type != "mandatory_rented":
                continue
            if len(trip.legs) != 1:
                errors.append(
                    self._issue(
                        "mandatory_route_mismatch",
                        "Zorunlu kiralık trip tam olarak bir sabit O-D bacağı olmalıdır.",
                        vehicle_id=trip.vehicle_id,
                        event_date=self._trip_start_date(trip),
                        trip_id=trip.trip_id,
                        related_values={"leg_count": len(trip.legs)},
                    )
                )
                continue
            leg = trip.legs[0]
            route_key = (leg.origin, leg.destination, trip.vehicle_type)
            if route_key not in allowed_route_keys:
                errors.append(
                    self._issue(
                        "mandatory_route_mismatch",
                        "Zorunlu kiralık trip statik rota/tür listesinde bulunmuyor.",
                        vehicle_id=trip.vehicle_id,
                        event_date=leg.departure_at.date(),
                        trip_id=trip.trip_id,
                        leg_id=leg.leg_id,
                        related_values={
                            "origin": leg.origin,
                            "destination": leg.destination,
                            "vehicle_type": trip.vehicle_type,
                        },
                    )
                )
            key = (
                leg.departure_at.date(),
                leg.origin,
                leg.destination,
                trip.vehicle_type,
            )
            actual[key] += 1
            actual_vehicle_ids[key].append(trip.vehicle_id)
            if completion_at is not None and self._trip_start_at(trip) > completion_at:
                errors.append(
                    self._issue(
                        "mandatory_departure_after_completion",
                        "Tüm talepler tamamlandıktan sonra yeni zorunlu sefer başlamış.",
                        vehicle_id=trip.vehicle_id,
                        event_date=leg.departure_at.date(),
                        trip_id=trip.trip_id,
                        leg_id=leg.leg_id,
                        related_values={
                            "completion_at": completion_at.isoformat(),
                            "trip_start_at": self._trip_start_at(trip).isoformat(),
                        },
                    )
                )
        return errors, actual, actual_vehicle_ids

    def _check_mandatory_daily_counts(
        self,
        required_routes: tuple[MandatoryRentalRoute, ...],
        actual: Counter[tuple[date, str, str, str]],
        actual_vehicle_ids: dict[tuple[date, str, str, str], list[str]],
        end_date: date,
    ) -> tuple[list[ValidationIssue], dict[str, Any]]:
        """Compare actual mandatory route counts with every required day."""

        errors: list[ValidationIssue] = []
        missing_count = 0
        excess_count = 0
        post_window_missing = 0
        current_date = self.policy.mandatory_schedule_start_date
        while current_date <= end_date:
            for route in required_routes:
                key = (
                    current_date,
                    route.origin,
                    route.destination,
                    route.vehicle_type,
                )
                found = actual.get(key, 0)
                if found < route.vehicle_count:
                    missing = route.vehicle_count - found
                    missing_count += missing
                    if current_date > self.policy.mandatory_forecast_end_date:
                        post_window_missing += missing
                    errors.append(
                        self._issue(
                            "mandatory_vehicle_missing",
                            "Günlük zorunlu kiralık araç sayısı eksik.",
                            event_date=current_date,
                            related_values={
                                "origin": route.origin,
                                "destination": route.destination,
                                "vehicle_type": route.vehicle_type,
                                "required_count": route.vehicle_count,
                                "actual_count": found,
                                "missing_count": missing,
                                "post_forecast_window": (
                                    current_date
                                    > self.policy.mandatory_forecast_end_date
                                ),
                            },
                        )
                    )
                elif found > route.vehicle_count:
                    excess = found - route.vehicle_count
                    excess_count += excess
                    errors.append(
                        self._issue(
                            "mandatory_vehicle_excess",
                            "Günlük zorunlu kiralık araç sayısı verilen adedi aşıyor.",
                            event_date=current_date,
                            related_values={
                                "origin": route.origin,
                                "destination": route.destination,
                                "vehicle_type": route.vehicle_type,
                                "required_count": route.vehicle_count,
                                "actual_count": found,
                                "excess_count": excess,
                                "vehicle_ids": actual_vehicle_ids[key],
                            },
                        )
                    )
            current_date += timedelta(days=1)

        return errors, {
            "mandatory_schedule_end_date": end_date.isoformat(),
            "mandatory_missing_count": missing_count,
            "mandatory_excess_count": excess_count,
            "mandatory_post_window_missing_count": post_window_missing,
        }

    def _check_cost_breakdown(
        self,
        reported: CostBreakdown | None,
        expected: CostBreakdown,
    ) -> tuple[list[ValidationIssue], list[ValidationIssue], dict[str, Any]]:
        """Compare reported costs, ceil values and duplicate cost units."""

        errors: list[ValidationIssue] = []
        warnings: list[ValidationIssue] = []
        metrics = {
            "computed_vehicle_cost_tl": expected.vehicle_cost_tl,
            "computed_sla_penalty_tl": expected.sla_penalty_tl,
            "computed_total_cost_tl": expected.total_cost_tl,
            "vehicle_cost_unit_count": len(expected.vehicle_cost_lines),
            "duplicate_vehicle_cost_count": 0,
        }
        if reported is None:
            issue = self._issue(
                "reported_cost_missing",
                "Raporlanan CostBreakdown yok; maliyet yalnız yeniden hesaplandı.",
                related_values={
                    "computed_total_cost_tl": expected.total_cost_tl,
                },
            )
            if self.policy.reported_cost_breakdown_required:
                errors.append(issue)
            else:
                warnings.append(issue)
            return errors, warnings, metrics

        totals = (
            "mandatory_rented_hourly_cost_tl",
            "mandatory_rented_distance_cost_tl",
            "spot_hourly_cost_tl",
            "spot_distance_cost_tl",
            "sla_penalty_tl",
            "total_cost_tl",
        )
        for field_name in totals:
            actual_value = getattr(reported, field_name)
            expected_value = getattr(expected, field_name)
            if abs(actual_value - expected_value) > self.policy.cost_tolerance_tl:
                errors.append(
                    self._issue(
                        "cost_total_mismatch",
                        "Raporlanan maliyet bileşeni yeniden hesaplanan değerle uyuşmuyor.",
                        related_values={
                            "field": field_name,
                            "reported": actual_value,
                            "expected": expected_value,
                            "difference": actual_value - expected_value,
                        },
                    )
                )

        errors.extend(self._check_reported_vehicle_lines(reported, expected, metrics))
        errors.extend(self._check_reported_sla_lines(reported, expected))
        return errors, warnings, metrics

    def _check_reported_vehicle_lines(
        self,
        reported: CostBreakdown,
        expected: CostBreakdown,
        metrics: dict[str, Any],
    ) -> list[ValidationIssue]:
        """Validate ceil arithmetic and one-time physical vehicle charging."""

        errors = self._check_duplicate_vehicle_cost_lines(reported, metrics)
        expected_by_unit = {
            line.cost_unit_id: line for line in expected.vehicle_cost_lines
        }
        reported_by_unit = {
            line.cost_unit_id: line for line in reported.vehicle_cost_lines
        }
        for unit_id, expected_line in expected_by_unit.items():
            actual_line = reported_by_unit.get(unit_id)
            if actual_line is None:
                errors.append(
                    self._issue(
                        "vehicle_cost_unit_missing",
                        "Beklenen araç maliyet satırı raporda yok.",
                        vehicle_id=expected_line.vehicle_id,
                        related_values={"cost_unit_id": unit_id},
                    )
                )
                continue
            errors.extend(self._compare_vehicle_cost_line(actual_line, expected_line))
        for unit_id, actual_line in reported_by_unit.items():
            if unit_id not in expected_by_unit:
                errors.append(
                    self._issue(
                        "unexpected_vehicle_cost_unit",
                        "Plan aktivitesine karşılık gelmeyen araç maliyeti raporlanmış.",
                        vehicle_id=actual_line.vehicle_id,
                        related_values={"cost_unit_id": unit_id},
                    )
                )
        return errors

    def _check_duplicate_vehicle_cost_lines(
        self,
        reported: CostBreakdown,
        metrics: dict[str, Any],
    ) -> list[ValidationIssue]:
        """Report duplicate cost-unit and physical-vehicle line identities."""

        errors: list[ValidationIssue] = []
        unit_counts = Counter(line.cost_unit_id for line in reported.vehicle_cost_lines)
        for unit_id, count in unit_counts.items():
            if count > 1:
                errors.append(
                    self._issue(
                        "duplicate_vehicle_cost_unit",
                        "Aynı araç maliyet birimi birden fazla kez raporlanmış.",
                        related_values={"cost_unit_id": unit_id, "count": count},
                    )
                )

        if self.policy.hourly_cost_granularity == "vehicle":
            vehicle_counts = Counter(
                line.vehicle_id for line in reported.vehicle_cost_lines
            )
            for vehicle_id, count in vehicle_counts.items():
                if count > 1:
                    metrics["duplicate_vehicle_cost_count"] += count - 1
                    errors.append(
                        self._issue(
                            "duplicate_physical_vehicle_cost",
                            "Aynı fiziksel araç maliyeti birden fazla kez sayılmış.",
                            vehicle_id=vehicle_id,
                            related_values={"reported_line_count": count},
                        )
                    )
        return errors

    def _compare_vehicle_cost_line(
        self,
        actual_line: VehicleCostLine,
        expected_line: VehicleCostLine,
    ) -> list[ValidationIssue]:
        """Compare one reported vehicle line with independent minute/km costs."""

        errors: list[ValidationIssue] = []
        expected_ceil = self._ceil_hours_from_fractional_hours(
            actual_line.usage_hours
        )
        if actual_line.charged_hours != expected_ceil:
            errors.append(
                self._issue(
                    "hourly_ceil_mismatch",
                    "Raporlanan ücretlendirilen saat ceil kuralıyla uyuşmuyor.",
                    vehicle_id=actual_line.vehicle_id,
                    related_values={
                        "cost_unit_id": actual_line.cost_unit_id,
                        "usage_hours": actual_line.usage_hours,
                        "reported_charged_hours": actual_line.charged_hours,
                        "expected_charged_hours": expected_ceil,
                    },
                )
            )
        for field_name in (
            "usage_hours",
            "charged_hours",
            "distance_km",
            "hourly_cost_tl",
            "distance_cost_tl",
            "total_cost_tl",
        ):
            actual_value = getattr(actual_line, field_name)
            expected_value = getattr(expected_line, field_name)
            if abs(actual_value - expected_value) <= self.policy.cost_tolerance_tl:
                continue
            errors.append(
                self._issue(
                    "vehicle_cost_line_mismatch",
                    "Araç maliyet satırı bağımsız hesapla uyuşmuyor.",
                    vehicle_id=actual_line.vehicle_id,
                    related_values={
                        "cost_unit_id": actual_line.cost_unit_id,
                        "field": field_name,
                        "reported": actual_value,
                        "expected": expected_value,
                    },
                )
            )
        return errors

    def _check_reported_sla_lines(
        self,
        reported: CostBreakdown,
        expected: CostBreakdown,
    ) -> list[ValidationIssue]:
        """Compare reported per-part SLA lines with independent expected lines."""

        errors: list[ValidationIssue] = []
        expected_by_key = {
            (line.demand_part_id, line.delivered_at): line
            for line in expected.sla_penalty_lines
        }
        reported_by_key = {
            (line.demand_part_id, line.delivered_at): line
            for line in reported.sla_penalty_lines
        }
        key_counts = Counter(
            (line.demand_part_id, line.delivered_at)
            for line in reported.sla_penalty_lines
        )
        for key, count in key_counts.items():
            if count > 1:
                errors.append(
                    self._issue(
                        "duplicate_sla_cost_line",
                        "Aynı demand-part teslimatı için SLA satırı tekrarlanmış.",
                        event_date=key[1].date(),
                        related_values={"part_id": key[0], "count": count},
                    )
                )
        for line in reported.sla_penalty_lines:
            errors.extend(self._check_sla_line_formula(line))
        for key, expected_line in expected_by_key.items():
            actual_line = reported_by_key.get(key)
            if actual_line is None:
                errors.append(
                    self._issue(
                        "sla_cost_line_missing",
                        "Beklenen demand-part SLA satırı raporda yok.",
                        demand_id=expected_line.demand_id,
                        event_date=expected_line.delivered_at.date(),
                        related_values={
                            "part_id": expected_line.demand_part_id,
                            "delivered_at": expected_line.delivered_at.isoformat(),
                        },
                    )
                )
                continue
            errors.extend(self._compare_one_sla_line(actual_line, expected_line))
        for key, actual_line in reported_by_key.items():
            if key not in expected_by_key:
                errors.append(
                    self._issue(
                        "unexpected_sla_cost_line",
                        "Nihai teslimata karşılık gelmeyen SLA satırı raporlanmış.",
                        demand_id=actual_line.demand_id,
                        event_date=actual_line.delivered_at.date(),
                        related_values={"part_id": actual_line.demand_part_id},
                    )
                )
        return errors

    def _check_sla_line_formula(
        self,
        line: SlaPenaltyLine,
    ) -> list[ValidationIssue]:
        """Validate one reported part's minute ceil, desi and TL arithmetic."""

        delay_minutes = max(
            0.0,
            (line.delivered_at - line.due_at).total_seconds()
            / SECONDS_PER_MINUTE,
        )
        delay_hours = delay_minutes / MINUTES_PER_HOUR
        charged_hours = math.ceil(delay_minutes / MINUTES_PER_HOUR)
        penalty_tl = (
            line.delivered_desi
            * charged_hours
            * self.policy.sla_penalty_per_desi_hour_tl
        )
        valid = (
            abs(line.delay_hours - delay_hours)
            <= self.policy.travel_time_tolerance_minutes / MINUTES_PER_HOUR
            and line.charged_delay_hours == charged_hours
            and abs(line.penalty_tl - penalty_tl) <= self.policy.cost_tolerance_tl
        )
        if valid:
            return []
        return [
            self._issue(
                "sla_penalty_mismatch",
                "Raporlanan SLA gecikmesi, ceil saati veya cezası hatalı.",
                demand_id=line.demand_id,
                event_date=line.delivered_at.date(),
                related_values={
                    "part_id": line.demand_part_id,
                    "reported_delay_hours": line.delay_hours,
                    "expected_delay_hours": delay_hours,
                    "reported_charged_hours": line.charged_delay_hours,
                    "expected_charged_hours": charged_hours,
                    "reported_penalty_tl": line.penalty_tl,
                    "expected_penalty_tl": penalty_tl,
                },
            )
        ]

    def _compare_one_sla_line(
        self,
        actual: SlaPenaltyLine,
        expected: SlaPenaltyLine,
    ) -> list[ValidationIssue]:
        """Report any desi, deadline, delay-hour or TL mismatch on one SLA line."""

        numeric_fields = (
            "delivered_desi",
            "delay_hours",
            "charged_delay_hours",
            "penalty_tl",
        )
        mismatches = {
            field_name: {
                "reported": getattr(actual, field_name),
                "expected": getattr(expected, field_name),
            }
            for field_name in numeric_fields
            if abs(getattr(actual, field_name) - getattr(expected, field_name))
            > self.policy.cost_tolerance_tl
        }
        if actual.due_at != expected.due_at:
            mismatches["due_at"] = {
                "reported": actual.due_at.isoformat(),
                "expected": expected.due_at.isoformat(),
            }
        if not mismatches:
            return []
        return [
            self._issue(
                "sla_cost_line_mismatch",
                "Raporlanan SLA satırı bağımsız beklenen değerle uyuşmuyor.",
                demand_id=actual.demand_id,
                event_date=actual.delivered_at.date(),
                related_values={
                    "part_id": actual.demand_part_id,
                    "mismatches": mismatches,
                },
            )
        ]

    def _latest_plan_activity(self, plan: TransportPlan) -> datetime | None:
        """Return the latest leg or handling timestamp in the plan."""

        candidates = [
            timestamp
            for trip in plan.trips
            for timestamp in (
                *(leg.arrival_at for leg in trip.legs),
                *(operation.end_at for operation in trip.handling_operations),
            )
        ]
        return max(candidates, default=None)

    @staticmethod
    def _trip_activity_bounds(trip: VehicleTrip) -> tuple[datetime, datetime]:
        """Return one trip's handling/travel envelope for overlap checks."""

        starts = [leg.departure_at for leg in trip.legs]
        ends = [leg.arrival_at for leg in trip.legs]
        starts.extend(item.start_at for item in trip.handling_operations)
        ends.extend(item.end_at for item in trip.handling_operations)
        return min(starts), max(ends)

    @staticmethod
    def _ceil_hours_from_fractional_hours(usage_hours: float) -> int:
        """Independently apply ceil after converting hours back to minutes."""

        usage_minutes = usage_hours * MINUTES_PER_HOUR
        return math.ceil(usage_minutes / MINUTES_PER_HOUR)

    @staticmethod
    def _trip_start_at(trip: VehicleTrip) -> datetime:
        """Return the earliest declared activity of a non-empty trip."""

        candidates = [leg.departure_at for leg in trip.legs]
        candidates.extend(
            operation.start_at for operation in trip.handling_operations
        )
        return min(candidates)

    @staticmethod
    def _trip_start_date(trip: VehicleTrip) -> date | None:
        """Return a trip date for issue context, if the trip has activity."""

        candidates = [leg.departure_at for leg in trip.legs]
        candidates.extend(
            operation.start_at for operation in trip.handling_operations
        )
        return min(candidates).date() if candidates else None

    @staticmethod
    def _issue(
        code: str,
        message: str,
        *,
        demand_id: str | None = None,
        vehicle_id: str | None = None,
        event_date: date | None = None,
        trip_id: str | None = None,
        leg_id: str | None = None,
        related_values: dict[str, Any] | None = None,
    ) -> ValidationIssue:
        """Create a consistently structured issue for report consumers."""

        return ValidationIssue(
            code=code,
            message=message,
            demand_id=demand_id,
            vehicle_id=vehicle_id,
            event_date=event_date,
            trip_id=trip_id,
            leg_id=leg_id,
            related_values=related_values or {},
        )


__all__ = ["TransportPlanValidator"]
