"""Independent cost calculation for a fully described transport plan.

The calculator consumes only domain dataclasses and an
:class:`~src.evaluation.policy.EvaluationPolicy`; it never loads Excel files
and never creates routes.  Vehicle activity is grouped by a physical
``vehicle_id`` by default, so carrying two demands on one vehicle does not
duplicate that vehicle's cost.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from typing import Iterable

from src.domain import ProblemData, TravelArc, VehicleType
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


class CostCalculationError(ValueError):
    """Report structural input that prevents deterministic TL calculation.

    ``CostCalculator`` and its public rounding helpers raise this error for
    unknown master data, negative/non-finite minutes, overlapping physical
    vehicle use or reused mandatory identities.  Validator callers do not rely
    on this exception because their independent audit converts failures to
    ``ValidationIssue`` records.
    """


@dataclass(frozen=True, slots=True)
class _VehicleCostUnit:
    """Internal grouping to which hourly ceil is applied exactly once."""

    cost_unit_id: str
    vehicle_id: str
    vehicle_type: str
    ownership_type: VehicleOwnershipType
    trips: tuple[VehicleTrip, ...]
    legs: tuple[VehicleLeg, ...]
    usage_minutes: float


def ceil_billable_minutes(usage_minutes: float) -> int:
    """Convert non-negative usage minutes to billed whole hours.

    Args:
        usage_minutes: Total physical vehicle activity in minutes.

    Returns:
        ``ceil(usage_minutes / 60 minutes/hour)`` as an integer hour count.

    Connected code and rules:
        ``CostCalculator`` calls this function once per config-selected cost
        unit.  Exact 16 hours (960 minutes) remains 16, while 961 minutes is
        billed as 17 hours.  No date-window filtering is performed.

    Raises:
        CostCalculationError: If minutes are negative or non-finite.
    """

    if not math.isfinite(usage_minutes):
        raise CostCalculationError("Araç kullanım dakikası sonlu olmalıdır.")
    if usage_minutes < 0:
        raise CostCalculationError("Araç kullanım dakikası negatif olamaz.")
    return math.ceil(usage_minutes / MINUTES_PER_HOUR)


def ceil_billable_hours(usage_hours: float) -> int:
    """Compatibility wrapper that bills a fractional number of hours.

    Args:
        usage_hours: Non-negative physical usage measured in hours.

    Returns:
        Whole billed hours after converting the input to minutes and applying
        :func:`ceil_billable_minutes`.

    Connected code and errors:
        Existing callers may use this public helper, but ``CostCalculator``
        itself works in minutes to mirror the competition formula exactly.
        Negative and non-finite values raise :class:`CostCalculationError`.
    """

    if not math.isfinite(usage_hours):
        raise CostCalculationError("Araç kullanım süresi sonlu olmalıdır.")
    if usage_hours < 0:
        raise CostCalculationError("Araç kullanım süresi negatif olamaz.")
    return ceil_billable_minutes(usage_hours * MINUTES_PER_HOUR)


class CostCalculator:
    """Calculate vehicle, kilometre and per-part SLA costs for a plan.

    The caller supplies source-independent problem/plan models and an
    ``EvaluationPolicy``; the calculator returns ``CostBreakdown`` and never
    loads Excel or invokes planning code.  Unknown master data, overlapping
    use of one physical vehicle, mandatory ID reuse and invalid durations are
    reported as ``CostCalculationError``.
    """

    def __init__(
        self,
        problem: ProblemData,
        policy: EvaluationPolicy,
    ) -> None:
        """Index immutable problem data used repeatedly by cost formulas."""

        self.problem = problem
        self.policy = policy
        self._vehicle_types = {
            vehicle.name: vehicle for vehicle in problem.static.vehicle_types
        }
        self._arcs = {
            (arc.origin, arc.destination): arc
            for arc in problem.static.travel_arcs
        }
        self._demands = {
            demand.demand_id: demand for demand in problem.demands
        }

    def calculate(self, plan: TransportPlan) -> CostBreakdown:
        """Return a complete TL breakdown without mutating ``plan``.

        Vehicle usage includes linked handling, travel, intermediate waits and
        the final unload.  The hour formula is
        ``ceil(total minutes / 60) × TL/hour`` and distance is
        ``total km × TL/km``.  SLA is calculated per delivered part in desi.
        Structural inconsistencies raise ``CostCalculationError``.
        """

        self._validate_physical_vehicle_usage(plan.trips)

        vehicle_lines = tuple(
            self._calculate_vehicle_line(unit)
            for unit in self._build_vehicle_cost_units(plan)
        )
        sla_lines = tuple(self._calculate_sla_lines(plan))

        # Aggregation remains separate by ownership and rate type so a caller
        # can audit mandatory-rented, spot, distance and hourly components.
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
        total = (
            mandatory_hourly
            + mandatory_distance
            + spot_hourly
            + spot_distance
            + sla_penalty
        )

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

    def _build_vehicle_cost_units(
        self,
        plan: TransportPlan,
    ) -> tuple[_VehicleCostUnit, ...]:
        """Group activity at the config-selected ceil granularity."""

        granularity = self.policy.hourly_cost_granularity
        if granularity == "vehicle":
            return self._vehicle_level_units(plan.trips)
        if granularity == "trip":
            return tuple(self._trip_level_unit(trip) for trip in plan.trips)
        return tuple(
            self._leg_level_unit(trip, leg)
            for trip in plan.trips
            for leg in trip.legs
        )

    def _vehicle_level_units(
        self,
        trips: Iterable[VehicleTrip],
    ) -> tuple[_VehicleCostUnit, ...]:
        """Create one hourly cost unit for each physical vehicle ID."""

        grouped: dict[str, list[VehicleTrip]] = defaultdict(list)
        for trip in trips:
            grouped[trip.vehicle_id].append(trip)

        units: list[_VehicleCostUnit] = []
        for vehicle_id, vehicle_trips in sorted(grouped.items()):
            ordered_trips = tuple(
                sorted(vehicle_trips, key=lambda item: self._trip_bounds(item)[0])
            )
            self._ensure_consistent_vehicle_identity(ordered_trips)
            all_legs = tuple(leg for trip in ordered_trips for leg in trip.legs)

            # The current documented assumption uses the full activity
            # envelope and therefore includes idle time between activities.
            if self.policy.include_idle_between_activities:
                starts_and_ends = [self._trip_bounds(trip) for trip in ordered_trips]
                usage_minutes = self._duration_minutes(
                    min(bounds[0] for bounds in starts_and_ends),
                    max(bounds[1] for bounds in starts_and_ends),
                )
            else:
                usage_minutes = math.fsum(
                    self._duration_minutes(*self._trip_bounds(trip))
                    for trip in ordered_trips
                )

            representative = ordered_trips[0]
            units.append(
                _VehicleCostUnit(
                    cost_unit_id=f"vehicle:{vehicle_id}",
                    vehicle_id=vehicle_id,
                    vehicle_type=representative.vehicle_type,
                    ownership_type=representative.ownership_type,
                    trips=ordered_trips,
                    legs=all_legs,
                    usage_minutes=usage_minutes,
                )
            )
        return tuple(units)

    def _trip_level_unit(self, trip: VehicleTrip) -> _VehicleCostUnit:
        """Create one hourly cost unit for one declared vehicle trip."""

        usage_minutes = self._duration_minutes(*self._trip_bounds(trip))
        return _VehicleCostUnit(
            cost_unit_id=f"trip:{trip.trip_id}",
            vehicle_id=trip.vehicle_id,
            vehicle_type=trip.vehicle_type,
            ownership_type=trip.ownership_type,
            trips=(trip,),
            legs=trip.legs,
            usage_minutes=usage_minutes,
        )

    def _leg_level_unit(
        self,
        trip: VehicleTrip,
        leg: VehicleLeg,
    ) -> _VehicleCostUnit:
        """Create one hourly cost unit for a leg plus linked handling."""

        usage_minutes = self._duration_minutes(*self._leg_bounds(trip, leg))
        return _VehicleCostUnit(
            cost_unit_id=f"leg:{leg.leg_id}",
            vehicle_id=trip.vehicle_id,
            vehicle_type=trip.vehicle_type,
            ownership_type=trip.ownership_type,
            trips=(trip,),
            legs=(leg,),
            usage_minutes=usage_minutes,
        )

    def _calculate_vehicle_line(
        self,
        unit: _VehicleCostUnit,
    ) -> VehicleCostLine:
        """Apply hourly ceil and distance rates once to one cost unit."""

        vehicle_type = self._vehicle_type_for(unit.vehicle_type)
        hourly_rate, distance_rate = self._rates_for(
            vehicle_type,
            unit.ownership_type,
        )
        distance_km = math.fsum(self._distance_for(leg) for leg in unit.legs)
        charged_hours = ceil_billable_minutes(unit.usage_minutes)
        hourly_cost = charged_hours * hourly_rate
        distance_cost = distance_km * distance_rate

        return VehicleCostLine(
            cost_unit_id=unit.cost_unit_id,
            granularity=self.policy.hourly_cost_granularity,
            vehicle_id=unit.vehicle_id,
            ownership_type=unit.ownership_type,
            trip_ids=tuple(trip.trip_id for trip in unit.trips),
            leg_ids=tuple(leg.leg_id for leg in unit.legs),
            usage_hours=unit.usage_minutes / MINUTES_PER_HOUR,
            charged_hours=charged_hours,
            distance_km=distance_km,
            hourly_cost_tl=hourly_cost,
            distance_cost_tl=distance_cost,
            total_cost_tl=hourly_cost + distance_cost,
        )

    def _calculate_sla_lines(
        self,
        plan: TransportPlan,
    ) -> list[SlaPenaltyLine]:
        """Calculate penalty independently for every final part delivery."""

        parts = {part.part_id: part for part in plan.demand_parts}
        lines: list[SlaPenaltyLine] = []
        for trip in plan.trips:
            operations = {
                operation.operation_id: operation
                for operation in trip.handling_operations
            }
            for leg in trip.legs:
                for assignment in leg.assignments:
                    if not assignment.is_final_delivery:
                        continue
                    part = parts.get(assignment.demand_part_id)
                    if part is None:
                        raise CostCalculationError(
                            "SLA hesabındaki assignment bilinmeyen demand part "
                            f"kullanıyor: {assignment.demand_part_id!r}"
                        )
                    demand = self._demands.get(part.demand_id)
                    if demand is None:
                        raise CostCalculationError(
                            f"SLA hesabında bilinmeyen demand_id: {part.demand_id!r}"
                        )

                    # SLA ends after final unloading.  If the operation is
                    # missing, arrival is used only so a cost can still be
                    # computed; the validator reports the missing operation.
                    unload = operations.get(leg.unloading_operation_id or "")
                    delivered_at = unload.end_at if unload is not None else leg.arrival_at
                    arc = self._arc_for(demand.origin, demand.destination)
                    due_at = datetime.combine(
                        demand.demand_date,
                        demand.time_slot,
                    ) + self._days_as_timedelta(arc.target_delivery_days)
                    delay_minutes = max(
                        0.0,
                        self._duration_minutes(due_at, delivered_at),
                    )
                    charged_delay = ceil_billable_minutes(delay_minutes)
                    penalty = (
                        assignment.desi
                        * charged_delay
                        * self.policy.sla_penalty_per_desi_hour_tl
                    )
                    lines.append(
                        SlaPenaltyLine(
                            demand_id=demand.demand_id,
                            demand_part_id=part.part_id,
                            delivered_desi=assignment.desi,
                            due_at=due_at,
                            delivered_at=delivered_at,
                            delay_hours=delay_minutes / MINUTES_PER_HOUR,
                            charged_delay_hours=charged_delay,
                            penalty_tl=penalty,
                        )
                    )
        return lines

    def _trip_bounds(self, trip: VehicleTrip) -> tuple[datetime, datetime]:
        """Return earliest and latest activity for a declared trip."""

        starts = [leg.departure_at for leg in trip.legs]
        ends = [leg.arrival_at for leg in trip.legs]
        starts.extend(operation.start_at for operation in trip.handling_operations)
        ends.extend(operation.end_at for operation in trip.handling_operations)
        if not starts or not ends:
            raise CostCalculationError(
                f"Araç seferinde maliyetlenecek aktivite yok: {trip.trip_id!r}"
            )
        return min(starts), max(ends)

    def _leg_bounds(
        self,
        trip: VehicleTrip,
        leg: VehicleLeg,
    ) -> tuple[datetime, datetime]:
        """Return travel plus the handling explicitly linked to one leg."""

        operations = {
            operation.operation_id: operation
            for operation in trip.handling_operations
        }
        start = leg.departure_at
        end = leg.arrival_at
        if leg.loading_operation_id in operations:
            start = min(start, operations[leg.loading_operation_id].start_at)
        if leg.unloading_operation_id in operations:
            end = max(end, operations[leg.unloading_operation_id].end_at)
        return start, end

    def _ensure_consistent_vehicle_identity(
        self,
        trips: tuple[VehicleTrip, ...],
    ) -> None:
        """Prevent one physical ID from silently changing type or ownership."""

        identities = {
            (trip.vehicle_type, trip.ownership_type) for trip in trips
        }
        if len(identities) != 1:
            raise CostCalculationError(
                "Aynı vehicle_id farklı araç türü veya sahiplik türüyle "
                f"kullanılmış: {trips[0].vehicle_id!r}"
            )

    def _validate_physical_vehicle_usage(
        self,
        trips: tuple[VehicleTrip, ...],
    ) -> None:
        """Reject ID reuse that would merge or undercount physical cost.

        A mandatory daily vehicle/sefer is a separate physical identity, so a
        mandatory ``vehicle_id`` may occur in only one trip.  Spot vehicles may
        perform multiple non-overlapping trips; their idle interval is included
        only when the configured vehicle-level activity envelope requires it.
        """

        grouped: dict[str, list[VehicleTrip]] = defaultdict(list)
        for trip in trips:
            grouped[trip.vehicle_id].append(trip)

        for vehicle_id, vehicle_trips in grouped.items():
            ordered = tuple(
                sorted(vehicle_trips, key=lambda item: self._trip_bounds(item)[0])
            )
            self._ensure_consistent_vehicle_identity(ordered)
            if (
                ordered[0].ownership_type == "mandatory_rented"
                and len(ordered) > 1
            ):
                raise CostCalculationError(
                    "Her günlük zorunlu kiralık ayrı vehicle_id kullanmalıdır: "
                    f"{vehicle_id!r}"
                )
            for previous, following in zip(ordered, ordered[1:], strict=False):
                previous_end = self._trip_bounds(previous)[1]
                following_start = self._trip_bounds(following)[0]
                if following_start < previous_end:
                    raise CostCalculationError(
                        "Aynı fiziksel aracın trip zamanları çakışıyor: "
                        f"{vehicle_id!r}"
                    )

    def _vehicle_type_for(self, vehicle_type: str) -> VehicleType:
        """Resolve a plan vehicle type against static master data."""

        try:
            return self._vehicle_types[vehicle_type]
        except KeyError as exc:
            raise CostCalculationError(
                f"Bilinmeyen araç türü: {vehicle_type!r}"
            ) from exc

    @staticmethod
    def _rates_for(
        vehicle_type: VehicleType,
        ownership_type: VehicleOwnershipType,
    ) -> tuple[float, float]:
        """Select mandatory-rented or spot hourly and kilometre rates."""

        if ownership_type == "mandatory_rented":
            return (
                vehicle_type.rented_hourly_cost_tl,
                vehicle_type.rented_cost_per_km_tl,
            )
        if ownership_type == "spot":
            return (
                vehicle_type.spot_hourly_cost_tl,
                vehicle_type.spot_cost_per_km_tl,
            )
        raise CostCalculationError(
            f"Bilinmeyen araç sahiplik türü: {ownership_type!r}"
        )

    def _distance_for(self, leg: VehicleLeg) -> float:
        """Return the matrix distance for one physical leg."""

        return self._arc_for(leg.origin, leg.destination).distance_km

    def _arc_for(self, origin: str, destination: str) -> TravelArc:
        """Resolve one directed matrix arc or explain why cost is impossible."""

        try:
            return self._arcs[(origin, destination)]
        except KeyError as exc:
            raise CostCalculationError(
                f"Mesafe/seyahat matrisi yayı bulunamadı: {origin!r} -> {destination!r}"
            ) from exc

    @staticmethod
    def _duration_minutes(start_at: datetime, end_at: datetime) -> float:
        """Convert a datetime interval to minutes for the billing formula."""

        return (end_at - start_at).total_seconds() / SECONDS_PER_MINUTE

    @staticmethod
    def _days_as_timedelta(days: int) -> timedelta:
        """Convert an integer SLA day count without duplicating date math."""

        return timedelta(days=days)


__all__ = [
    "CostCalculationError",
    "CostCalculator",
    "ceil_billable_hours",
    "ceil_billable_minutes",
]
