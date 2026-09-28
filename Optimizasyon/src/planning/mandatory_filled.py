"""Fill fixed mandatory rentals before same-O-D consolidated spot transport.

``GreedyMandatoryRentalFilledBuilder`` is a strategy-local greedy baseline.  It
preserves existing atomic parts, tries same-date/exact-O-D cargo on mandatory
vehicles, and delegates every unassigned part to the existing same-O-D spot
consolidation.  It is not an optimizer and introduces no multi-stop route,
transfer, vehicle reuse or exporter behaviour.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
import math
from time import perf_counter

from src.domain import (
    DemandAssignment,
    DemandPart,
    HandlingOperation,
    MandatoryRentalRoute,
    TransportPlan,
    VehicleLeg,
    VehicleTrip,
)
from src.evaluation import CostCalculator, TransportPlanValidator

from .baseline import BaselineBuildResult
from .consolidated import GreedySameODConsolidatedBuilder
from .handling_ledger import HandlingCapacityLedger
from .partitioning import partition_demands


class GreedyMandatoryRentalFilledBuilder(GreedySameODConsolidatedBuilder):
    """Fill mandatory fixed-route capacity, then consolidate remaining spot load.

    Args:
        problem: Standard static data and unchanged demand collection.
        policy: Shared tolerance, handling, costing and validation policy.

    Returns and connected layers:
        Inherited :meth:`build` returns ``BaselineBuildResult``.  This builder
        uses ``partition_demands`` once, a shared ``HandlingCapacityLedger``,
        inherited same-O-D FFD spot construction, ``CostCalculator`` and the
        independent ``TransportPlanValidator``.

    Strategy rules and failures:
        A part may enter a mandatory box only on its own demand date and exact
        fixed O-D.  Tır loads are accepted only when travel still arrives on
        that service date, preserving the existing quota-day pattern; this is
        a local safety preference, not a global competition rule.  Rejected
        parts remain in the spot pool.  Structural, cost or validation defects
        use inherited ``BaselineBuildError`` handling and no partial plan is
        returned.
    """

    def _build_result(self, started_at: float) -> BaselineBuildResult:
        """Build mandatory-first, remaining-spot, costed and validated result."""

        ordered_demands = tuple(
            sorted(
                self.problem.demands,
                key=lambda demand: (
                    datetime.combine(demand.demand_date, demand.time_slot),
                    demand.demand_id,
                ),
            )
        )
        parts = partition_demands(
            ordered_demands,
            self._maximum_non_truck_capacity(),
            self.policy.desi_tolerance,
        )
        ledger = HandlingCapacityLedger(
            self.problem.static.handling_capacities,
            self.policy.handling_minutes_per_desi,
            self.policy.desi_tolerance,
        )
        forecast_mandatory, assigned_ids = self._build_forecast_mandatory(
            parts,
            ledger,
        )
        remaining = tuple(part for part in parts if part.part_id not in assigned_ids)
        spot_trips = self._build_spot_trips(remaining, ledger)
        completion_at = self._spot_completion_at(forecast_mandatory + spot_trips)
        continuation = self._build_empty_continuation(completion_at)
        mandatory_trips = forecast_mandatory + continuation
        draft = TransportPlan(parts, mandatory_trips + spot_trips)
        cost = CostCalculator(self.problem, self.policy).calculate(draft)
        plan = TransportPlan(parts, draft.trips, reported_cost=cost)
        report = TransportPlanValidator(self.problem, self.policy).validate(plan)
        self._require_valid_report(report)
        return BaselineBuildResult(
            plan=plan,
            validation_report=report,
            total_cost_tl=cost.total_cost_tl,
            completion_at=completion_at,
            demand_part_count=len(parts),
            spot_vehicle_count=len(spot_trips),
            mandatory_vehicle_count=len(mandatory_trips),
            handling_usage=ledger.snapshot(),
            build_seconds=perf_counter() - started_at,
        )

    def _build_forecast_mandatory(
        self,
        parts: tuple[DemandPart, ...],
        ledger: HandlingCapacityLedger,
    ) -> tuple[tuple[VehicleTrip, ...], frozenset[str]]:
        """Build and greedily fill all mandatory vehicles in forecast dates."""

        routes = self._ordered_mandatory_routes()
        assigned_ids: set[str] = set()
        trips: list[VehicleTrip] = []
        service_date = self.policy.mandatory_schedule_start_date
        while service_date <= self.policy.mandatory_forecast_end_date:
            for route_index, route in enumerate(routes, start=1):
                candidates = self._route_candidates(
                    parts,
                    assigned_ids,
                    service_date,
                    route,
                )
                boxes = self._pack_mandatory_boxes(candidates, service_date, route)
                for sequence, box_parts in enumerate(boxes, start=1):
                    trip, newly_assigned = self._build_mandatory_box_trip(
                        service_date,
                        route_index,
                        sequence,
                        route,
                        box_parts,
                        ledger,
                    )
                    trips.append(trip)
                    assigned_ids.update(newly_assigned)
            service_date += timedelta(days=1)
        return tuple(trips), frozenset(assigned_ids)

    def _route_candidates(
        self,
        parts: tuple[DemandPart, ...],
        assigned_ids: set[str],
        service_date: date,
        route: MandatoryRentalRoute,
    ) -> tuple[DemandPart, ...]:
        """Return unassigned same-date/exact-O-D atoms in deterministic order."""

        candidates = [
            part
            for part in parts
            if part.part_id not in assigned_ids
            and self._part_matches_route(part, service_date, route)
        ]
        return tuple(
            sorted(
                candidates,
                key=lambda part: (
                    -part.desi,
                    self._ready_at(part),
                    part.part_id,
                ),
            )
        )

    def _pack_mandatory_boxes(
        self,
        candidates: tuple[DemandPart, ...],
        service_date: date,
        route: MandatoryRentalRoute,
    ) -> tuple[tuple[DemandPart, ...], ...]:
        """First-fit atoms into exactly the static mandatory vehicle count."""

        capacity = self._vehicle_types[route.vehicle_type].capacity_desi
        boxes: list[list[DemandPart]] = [
            [] for _ in range(route.vehicle_count)
        ]
        for part in candidates:
            for vehicle_box in boxes:
                proposed = (*vehicle_box, part)
                if self._mandatory_box_fits(
                    proposed,
                    capacity,
                    service_date,
                    route,
                ):
                    vehicle_box.append(part)
                    break
        return tuple(tuple(vehicle_box) for vehicle_box in boxes)

    def _mandatory_box_fits(
        self,
        parts: tuple[DemandPart, ...],
        capacity_desi: float,
        service_date: date,
        route: MandatoryRentalRoute,
    ) -> bool:
        """Check tolerance-aware capacity and service-day time feasibility."""

        total_desi = math.fsum(part.desi for part in parts)
        if total_desi > capacity_desi + self.policy.desi_tolerance:
            return False
        ready_at = max(self._ready_at(part) for part in parts)
        duration = timedelta(
            minutes=total_desi * self.policy.handling_minutes_per_desi
        )
        return ready_at + duration < self._loading_deadline(service_date, route)

    def _build_mandatory_box_trip(
        self,
        service_date: date,
        route_index: int,
        sequence: int,
        route: MandatoryRentalRoute,
        parts: tuple[DemandPart, ...],
        ledger: HandlingCapacityLedger,
    ) -> tuple[VehicleTrip, tuple[str, ...]]:
        """Build one loaded box or an empty trip when same-day load fails."""

        stem = self._mandatory_stem(service_date, route_index, sequence)
        if not parts:
            return self._empty_mandatory_trip(stem, service_date, route), ()
        total_desi = math.fsum(part.desi for part in parts)
        ready_at = max(self._ready_at(part) for part in parts)
        reservation = ledger.try_schedule_before(
            route.origin,
            ready_at,
            total_desi,
            self._loading_deadline(service_date, route),
        )
        if reservation is None:
            return self._empty_mandatory_trip(stem, service_date, route), ()
        trip = self._loaded_mandatory_trip(
            stem,
            route,
            parts,
            total_desi,
            reservation,
            ledger,
        )
        return trip, tuple(part.part_id for part in parts)

    def _loaded_mandatory_trip(
        self,
        stem: str,
        route: MandatoryRentalRoute,
        parts: tuple[DemandPart, ...],
        total_desi: float,
        load_interval: tuple[datetime, datetime],
        ledger: HandlingCapacityLedger,
    ) -> VehicleTrip:
        """Create grouped handling and one fixed leg for a loaded rental."""

        load_start, load_end = load_interval
        arc = self._arc_for(route.origin, route.destination)
        arrival_at = load_end + timedelta(
            hours=arc.travel_hours_for(route.vehicle_type)
        )
        unload_start, unload_end = ledger.schedule(
            route.destination,
            arrival_at,
            total_desi,
        )
        part_ids = tuple(part.part_id for part in parts)
        assignments = tuple(
            DemandAssignment(part.part_id, part.desi, is_final_delivery=True)
            for part in parts
        )
        load = HandlingOperation(
            f"{stem}-LOAD", stem, route.origin, "load", load_start, load_end,
            total_desi, part_ids,
        )
        unload = HandlingOperation(
            f"{stem}-UNLOAD", stem, route.destination, "unload",
            unload_start, unload_end, total_desi, part_ids,
        )
        leg = VehicleLeg(
            f"{stem}-LEG", route.origin, route.destination, load_end,
            arrival_at, assignments, load.operation_id, unload.operation_id,
        )
        return VehicleTrip(
            f"{stem}-TRIP", stem, route.vehicle_type, "mandatory_rented",
            (leg,), (load, unload),
        )

    def _empty_mandatory_trip(
        self,
        stem: str,
        service_date: date,
        route: MandatoryRentalRoute,
    ) -> VehicleTrip:
        """Create one normally costed fixed-route empty mandatory movement."""

        departure_at = self._mandatory_departure_at(service_date)
        arc = self._arc_for(route.origin, route.destination)
        arrival_at = departure_at + timedelta(
            hours=arc.travel_hours_for(route.vehicle_type)
        )
        leg = VehicleLeg(
            f"{stem}-LEG", route.origin, route.destination,
            departure_at, arrival_at,
        )
        return VehicleTrip(
            f"{stem}-TRIP", stem, route.vehicle_type, "mandatory_rented",
            (leg,),
        )

    def _build_empty_continuation(
        self,
        completion_at: datetime | None,
    ) -> tuple[VehicleTrip, ...]:
        """Continue exact mandatory counts empty after the forecast if needed."""

        if completion_at is None:
            return ()
        routes = self._ordered_mandatory_routes()
        service_date = self.policy.mandatory_forecast_end_date + timedelta(days=1)
        trips: list[VehicleTrip] = []
        while service_date <= completion_at.date():
            for route_index, route in enumerate(routes, start=1):
                for sequence in range(1, route.vehicle_count + 1):
                    stem = self._mandatory_stem(
                        service_date,
                        route_index,
                        sequence,
                    )
                    trips.append(
                        self._empty_mandatory_trip(stem, service_date, route)
                    )
            service_date += timedelta(days=1)
        return tuple(trips)

    def _loading_deadline(
        self,
        service_date: date,
        route: MandatoryRentalRoute,
    ) -> datetime:
        """Return exclusive load end preserving departure/arrival service day.

        The competition only requires a mandatory rental to **depart** within its
        service day (Q&A 8.1), so reserving the travel time is stricter than the
        rules demand.  It is kept for Tır routes whose destination has no spare
        daily quota, because it holds those arrivals to exactly one per day.

        Routes listed by ``_next_day_arrival_routes`` skip that margin: their
        destination has headroom, and the ledger already reserves the day where
        two of their arrivals could collide.  Without this, the entire 17:00
        demand slot on the long İstanbul lanes was rejected and rode paid spot
        vehicles instead.
        """

        next_midnight = datetime.combine(service_date + timedelta(days=1), time.min)
        if route.vehicle_type != self.policy.truck_vehicle_type:
            return next_midnight
        if route in self._next_day_arrival_routes():
            return next_midnight
        arc = self._arc_for(route.origin, route.destination)
        return next_midnight - timedelta(
            hours=arc.travel_hours_for(route.vehicle_type)
        )

    def _part_matches_route(
        self,
        part: DemandPart,
        service_date: date,
        route: MandatoryRentalRoute,
    ) -> bool:
        """Check the strategy's exact demand-date and fixed-O-D eligibility."""

        demand = self._demands[part.demand_id]
        return (
            demand.demand_date == service_date
            and demand.origin == route.origin
            and demand.destination == route.destination
        )

    def _ready_at(self, part: DemandPart) -> datetime:
        """Resolve one atom's unchanged parent-demand ready timestamp."""

        demand = self._demands[part.demand_id]
        return datetime.combine(demand.demand_date, demand.time_slot)

    def _ordered_mandatory_routes(self) -> tuple[MandatoryRentalRoute, ...]:
        """Return static routes in origin, destination and vehicle-type order."""

        return tuple(
            sorted(
                self.problem.static.mandatory_rental_routes,
                key=lambda route: (
                    route.origin,
                    route.destination,
                    route.vehicle_type,
                ),
            )
        )

    @staticmethod
    def _mandatory_stem(
        service_date: date,
        route_index: int,
        sequence: int,
    ) -> str:
        """Build one stable physical vehicle/trip/leg identifier stem."""

        return f"MFILL-{service_date:%Y%m%d}-{route_index:02d}-{sequence:02d}"


__all__ = ["GreedyMandatoryRentalFilledBuilder"]
