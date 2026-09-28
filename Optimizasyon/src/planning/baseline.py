"""Build a conservative direct plan and require independent validation.

The builder is intentionally a feasibility baseline, not a cost optimizer.  It
splits positive demands to the largest non-Tır capacity, assigns one new spot
vehicle to every atom, schedules shared daily handling, adds all empty
mandatory rentals, then invokes the existing calculator and validator.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import math
from time import perf_counter

from src.domain import (
    DemandAssignment,
    DemandInstance,
    DemandPart,
    HandlingOperation,
    MandatoryRentalRoute,
    ProblemData,
    TransportPlan,
    TravelArc,
    ValidationIssue,
    ValidationReport,
    VehicleLeg,
    VehicleTrip,
    VehicleType,
)
from src.evaluation import CostCalculator, EvaluationPolicy, TransportPlanValidator

from .handling_ledger import (
    HandlingCapacityError,
    HandlingCapacityLedger,
    HandlingUsage,
)
from .partitioning import DemandPartitionError, partition_demands
from .truck_ledger import DailyTruckVisitLedger, TruckQuotaError


MINUTES_PER_HOUR = 60.0
# These timestamps are construction choices for this conservative strategy;
# they are not registered as global competition rules or global config.
FIRST_MANDATORY_DEPARTURE_TIME = time(9, 0)
LATER_MANDATORY_DEPARTURE_TIME = time.min
# Free Tır visits a destination must keep before a mandatory rental bound for it
# is allowed to arrive the day after it departs.  Two, because such a route can
# put two of its own arrivals on one calendar day; see
# ``ConservativeDirectBaselineBuilder._next_day_arrival_routes``.
NEXT_DAY_ARRIVAL_SPARE_QUOTA = 2


class BaselineBuildError(RuntimeError):
    """Report a baseline construction or validation failure with error codes.

    ``ConservativeDirectBaselineBuilder.build`` raises this exception instead
    of returning a plan that failed independent validation.  ``error_codes`` supports CLI and
    test diagnostics; ``issues`` preserves the first validator records with
    demand, vehicle, date and related-value context.  No partial plan is saved.
    """

    def __init__(
        self,
        message: str,
        *,
        error_codes: tuple[str, ...] = (),
        issues: tuple[ValidationIssue, ...] = (),
    ) -> None:
        """Store stable machine-readable codes beside the human explanation."""

        super().__init__(message)
        self.error_codes = tuple(error_codes)
        self.issues = tuple(issues)


@dataclass(frozen=True, slots=True)
class BaselineBuildResult:
    """Return a validated baseline plan and transparent build diagnostics.

    Times are local naive ``datetime`` values consistent with input data;
    monetary totals are TL, handling usage is desi, and ``build_seconds`` is
    wall-clock seconds.  The CLI consumes this record and downstream code may
    use only its standard ``TransportPlan`` without depending on this builder.
    """

    plan: TransportPlan
    validation_report: ValidationReport
    total_cost_tl: float
    completion_at: datetime | None
    demand_part_count: int
    spot_vehicle_count: int
    mandatory_vehicle_count: int
    handling_usage: tuple[HandlingUsage, ...]
    build_seconds: float


class ConservativeDirectBaselineBuilder:
    """Construct a deterministic validator-passing direct transport plan.

    Args:
        problem: Static network/master data paired with standard demand rows.
        policy: Config-backed handling, costing and mandatory-schedule rules.

    Connected modules and rules:
        ``partitioning`` creates atomic parts and ``handling_ledger`` reserves
        centre/day capacity.  Every part receives a unique non-Tır spot vehicle
        and one direct leg.  Mandatory rentals remain empty.  The final plan is
        costed by ``CostCalculator`` and accepted only if
        ``TransportPlanValidator`` returns zero errors.  The class never reads
        Excel, forecasts, optimizes, consolidates or exports a plan workbook.
        These construction choices are strategy-local and must not be treated
        as global competition business rules.
    """

    def __init__(self, problem: ProblemData, policy: EvaluationPolicy) -> None:
        """Index immutable master data needed for deterministic construction."""

        self.problem = problem
        self.policy = policy
        self._demands = {
            demand.demand_id: demand for demand in problem.demands
        }
        self._arcs = {
            (arc.origin, arc.destination): arc
            for arc in problem.static.travel_arcs
        }
        self._vehicle_types = {
            vehicle.name: vehicle for vehicle in problem.static.vehicle_types
        }
        # Created per build so a reused builder never inherits stale quota.
        self._truck_ledger = self._new_truck_ledger()

    def _new_truck_ledger(self) -> DailyTruckVisitLedger:
        """Return a fresh Tır visit ledger pre-charged with mandatory rentals."""

        reserve: dict[str, int] = {}
        for route in self._next_day_arrival_routes():
            reserve[route.destination] = reserve.get(route.destination, 0) + 1
        return DailyTruckVisitLedger(
            self.problem.static.daily_truck_capacities,
            self.problem.static.mandatory_rental_routes,
            self.policy.truck_vehicle_type,
            extra_arrival_reserve=reserve,
        )

    def _next_day_arrival_routes(self) -> tuple[MandatoryRentalRoute, ...]:
        """Return mandatory Tır routes whose arrival may slip past midnight.

        Returns:
            Routes whose destination keeps at least
            ``NEXT_DAY_ARRIVAL_SPARE_QUOTA`` Tır visits free after the fixed
            mandatory schedule.

        Why a route is allowed to slip:
            Requiring a mandatory Tır to arrive on its departure day forces
            loading to finish a full travel time before midnight.  On the long
            İstanbul lanes that rejects the entire 17:00 demand slot, which then
            rides paid spot vehicles departing at the same time and arriving at
            the same time.  Letting the arrival fall on the next day recovers
            that cargo for free.

        Why the spare-quota gate exists:
            Departure time follows cargo readiness, so a relaxed route arrives
            same-day on some dates and next-day on others, and two of its
            arrivals can land on one calendar day.  At Balıkesir, whose quota is
            exactly one and fully used by its mandatory Tır, that produced a
            validator rejection.  Only destinations with real headroom are
            relaxed, and :meth:`_new_truck_ledger` reserves the collision day.
        """

        arrivals: dict[str, int] = {}
        for route in self.problem.static.mandatory_rental_routes:
            if route.vehicle_type != self.policy.truck_vehicle_type:
                continue
            arrivals[route.destination] = (
                arrivals.get(route.destination, 0) + route.vehicle_count
            )
        quotas = {
            item.center: item.max_daily_truck_visits
            for item in self.problem.static.daily_truck_capacities
        }
        return tuple(
            route
            for route in self.problem.static.mandatory_rental_routes
            if route.vehicle_type == self.policy.truck_vehicle_type
            and quotas.get(route.destination, 0)
            - arrivals.get(route.destination, 0)
            >= NEXT_DAY_ARRIVAL_SPARE_QUOTA
        )

    def _truck_capacity_desi(self) -> float:
        """Return the Tır capacity in desi, or zero when the type is unknown."""

        vehicle = self._vehicle_types.get(self.policy.truck_vehicle_type)
        return 0.0 if vehicle is None else vehicle.capacity_desi

    def build(self) -> BaselineBuildResult:
        """Build, cost and independently validate the conservative baseline.

        Returns:
            ``BaselineBuildResult`` containing a fully reported, valid plan,
            completion timestamp, vehicle counts, daily handling desi and
            elapsed wall-clock seconds.

        Raises:
            BaselineBuildError: For invalid/non-partitionable data, unavailable
                non-Tır capacity, impossible same-day handling, missing matrix
                data, mandatory quota/schedule violations, cost failures or any
                validator error.  The exception includes structured codes.
        """

        started_at = perf_counter()
        try:
            return self._build_result(started_at)
        except BaselineBuildError:
            raise
        except DemandPartitionError as exc:
            raise BaselineBuildError(
                str(exc), error_codes=("demand_partition_failed",)
            ) from exc
        except HandlingCapacityError as exc:
            raise BaselineBuildError(
                str(exc), error_codes=("handling_capacity_unavailable",)
            ) from exc
        except TruckQuotaError as exc:
            raise BaselineBuildError(
                str(exc), error_codes=("truck_quota_unavailable",)
            ) from exc
        except (KeyError, ValueError) as exc:
            raise BaselineBuildError(
                f"Başlangıç planı kurulamadı: {exc}",
                error_codes=("baseline_input_invalid",),
            ) from exc

    def _build_result(self, started_at: float) -> BaselineBuildResult:
        """Assemble the plan and package diagnostics after validation."""

        ordered_demands = tuple(sorted(self.problem.demands, key=_demand_sort_key))
        # Atoms stay within the largest non-Tır capacity on purpose.  A Tır-sized
        # atom bound for a centre with zero Tır quota (Bilecik, Kütahya, ...)
        # would have no vehicle able to carry it at all, making the whole plan
        # infeasible.  Keeping atoms Kamyon-sized guarantees a fallback, while
        # Tır is unlocked one level up where several atoms share one vehicle.
        maximum_part_desi = self._maximum_non_truck_capacity()
        parts = partition_demands(
            ordered_demands,
            maximum_part_desi,
            self.policy.desi_tolerance,
        )
        self._truck_ledger = self._new_truck_ledger()
        ledger = HandlingCapacityLedger(
            self.problem.static.handling_capacities,
            self.policy.handling_minutes_per_desi,
            self.policy.desi_tolerance,
        )
        spot_trips = self._build_spot_trips(parts, ledger)
        completion_at = self._spot_completion_at(spot_trips)
        schedule_end = self._mandatory_schedule_end(completion_at)
        mandatory_trips = self._build_mandatory_trips(schedule_end, completion_at)
        draft = TransportPlan(parts, spot_trips + mandatory_trips)
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

    def _maximum_non_truck_capacity(self) -> float:
        """Return the largest finite positive non-Tır capacity in desi."""

        capacities = [
            vehicle.capacity_desi
            for vehicle in self._vehicle_types.values()
            if vehicle.name != self.policy.truck_vehicle_type
            and math.isfinite(vehicle.capacity_desi)
            and vehicle.capacity_desi > 0
        ]
        if not capacities:
            raise BaselineBuildError(
                "Pozitif kapasiteli Tır dışı araç türü bulunamadı.",
                error_codes=("non_truck_vehicle_unavailable",),
            )
        return max(capacities)

    def _build_spot_trips(
        self,
        parts: tuple[DemandPart, ...],
        ledger: HandlingCapacityLedger,
    ) -> tuple[VehicleTrip, ...]:
        """Create one deterministic direct spot trip for each atomic part."""

        trips: list[VehicleTrip] = []
        for index, part in enumerate(parts, start=1):
            demand = self._demands[part.demand_id]
            trips.append(self._build_one_spot_trip(index, part, demand, ledger))
        return tuple(trips)

    def _build_one_spot_trip(
        self,
        index: int,
        part: DemandPart,
        demand: DemandInstance,
        ledger: HandlingCapacityLedger,
    ) -> VehicleTrip:
        """Schedule load, direct travel and final unload for one atom."""

        arc = self._arc_for(demand.origin, demand.destination)
        vehicle_type = self._select_spot_vehicle(part, arc.distance_km)
        ready_at = datetime.combine(demand.demand_date, demand.time_slot)
        load_start, load_end = ledger.schedule(demand.origin, ready_at, part.desi)
        travel_hours = arc.travel_hours_for(vehicle_type.name)
        arrival_at = load_end + timedelta(hours=travel_hours)
        unload_start, unload_end = ledger.schedule(
            demand.destination, arrival_at, part.desi
        )
        return self._spot_trip_models(
            index,
            part,
            demand,
            vehicle_type,
            load_start,
            load_end,
            arrival_at,
            unload_start,
            unload_end,
        )

    def _select_spot_vehicle(
        self,
        part: DemandPart,
        distance_km: float,
    ) -> VehicleType:
        """Choose minimum approximate TL cost, then capacity and type name."""

        candidates = [
            vehicle
            for vehicle in self._vehicle_types.values()
            if vehicle.name != self.policy.truck_vehicle_type
            and vehicle.capacity_desi + self.policy.desi_tolerance >= part.desi
        ]
        if not candidates:
            raise BaselineBuildError(
                f"{part.part_id!r} için yeterli Tır dışı spot araç yok.",
                error_codes=("non_truck_vehicle_unavailable",),
            )
        return min(
            candidates,
            key=lambda vehicle: (
                self._approximate_spot_cost(part, vehicle, distance_km),
                vehicle.capacity_desi,
                vehicle.name,
            ),
        )

    def _approximate_spot_cost(
        self,
        part: DemandPart,
        vehicle: VehicleType,
        distance_km: float,
    ) -> float:
        """Return direct-trip estimate in TL using minute-based hour ceil."""

        demand = self._demands[part.demand_id]
        arc = self._arc_for(demand.origin, demand.destination)
        handling_minutes = (
            2.0 * part.desi * self.policy.handling_minutes_per_desi
        )
        usage_minutes = (
            handling_minutes
            + arc.travel_hours_for(vehicle.name) * MINUTES_PER_HOUR
        )
        return (
            math.ceil(usage_minutes / MINUTES_PER_HOUR)
            * vehicle.spot_hourly_cost_tl
            + distance_km * vehicle.spot_cost_per_km_tl
        )

    def _spot_trip_models(
        self,
        index: int,
        part: DemandPart,
        demand: DemandInstance,
        vehicle_type: VehicleType,
        load_start: datetime,
        load_end: datetime,
        arrival_at: datetime,
        unload_start: datetime,
        unload_end: datetime,
    ) -> VehicleTrip:
        """Create linked domain models for one scheduled direct delivery."""

        stem = f"SPOT-{index:05d}"
        assignment = DemandAssignment(part.part_id, part.desi, is_final_delivery=True)
        load = HandlingOperation(
            f"{stem}-LOAD", stem, demand.origin, "load",
            load_start, load_end, part.desi, (part.part_id,),
        )
        unload = HandlingOperation(
            f"{stem}-UNLOAD", stem, demand.destination, "unload",
            unload_start, unload_end, part.desi, (part.part_id,),
        )
        leg = VehicleLeg(
            f"{stem}-LEG", demand.origin, demand.destination, load_end,
            arrival_at, (assignment,), load.operation_id, unload.operation_id,
        )
        return VehicleTrip(
            f"{stem}-TRIP", stem, vehicle_type.name, "spot", (leg,),
            (load, unload),
        )

    def _mandatory_schedule_end(self, completion_at: datetime | None) -> date:
        """Return inclusive max of forecast end and final delivery date."""

        completion_date = (
            completion_at.date()
            if completion_at is not None
            else self.policy.mandatory_forecast_end_date
        )
        return max(self.policy.mandatory_forecast_end_date, completion_date)

    def _build_mandatory_trips(
        self,
        end_date: date,
        completion_at: datetime | None,
    ) -> tuple[VehicleTrip, ...]:
        """Create every empty fixed-route rental through the inclusive horizon."""

        trips: list[VehicleTrip] = []
        routes = sorted(
            self.problem.static.mandatory_rental_routes,
            key=lambda route: (route.origin, route.destination, route.vehicle_type),
        )
        current = self.policy.mandatory_schedule_start_date
        while current <= end_date:
            departure_at = self._mandatory_departure_at(current)
            if completion_at is not None and departure_at > completion_at:
                raise BaselineBuildError(
                    "Zorunlu günlük kalkış son talep tesliminden sonraya düşüyor.",
                    error_codes=("mandatory_departure_after_completion",),
                )
            trips.extend(self._mandatory_trips_for_day(current, routes))
            current += timedelta(days=1)
        return tuple(trips)

    def _mandatory_trips_for_day(
        self,
        service_date: date,
        routes: list[MandatoryRentalRoute],
    ) -> list[VehicleTrip]:
        """Build exact static route/count movements for one service date."""

        trips: list[VehicleTrip] = []
        departure_at = self._mandatory_departure_at(service_date)
        for route_index, route in enumerate(routes, start=1):
            arc = self._arc_for(route.origin, route.destination)
            travel_hours = arc.travel_hours_for(route.vehicle_type)
            for sequence in range(1, route.vehicle_count + 1):
                stem = (
                    f"MAND-{service_date:%Y%m%d}-{route_index:02d}-{sequence:02d}"
                )
                leg = VehicleLeg(
                    f"{stem}-LEG", route.origin, route.destination,
                    departure_at, departure_at + timedelta(hours=travel_hours),
                )
                trips.append(
                    VehicleTrip(
                        f"{stem}-TRIP", stem, route.vehicle_type,
                        "mandatory_rented", (leg,),
                    )
                )
        return trips

    def _mandatory_departure_at(self, service_date: date) -> datetime:
        """Apply 09:00 on 29 June and 00:00 on every later service day."""

        departure_time = (
            FIRST_MANDATORY_DEPARTURE_TIME
            if service_date == self.policy.mandatory_schedule_start_date
            else LATER_MANDATORY_DEPARTURE_TIME
        )
        return datetime.combine(service_date, departure_time)

    def _arc_for(self, origin: str, destination: str) -> TravelArc:
        """Resolve one directed matrix arc or raise an explained build error."""

        try:
            return self._arcs[(origin, destination)]
        except KeyError as exc:
            raise BaselineBuildError(
                f"Mesafe/seyahat yayı bulunamadı: {origin!r}->{destination!r}.",
                error_codes=("missing_travel_arc",),
            ) from exc

    @staticmethod
    def _spot_completion_at(trips: tuple[VehicleTrip, ...]) -> datetime | None:
        """Return final spot unload completion; zero-demand plans return None."""

        return max(
            (
                operation.end_at
                for trip in trips
                for operation in trip.handling_operations
                if operation.operation_type == "unload"
            ),
            default=None,
        )

    @staticmethod
    def _require_valid_report(report: ValidationReport) -> None:
        """Raise with codes and first contexts instead of returning invalidity."""

        if report.is_valid:
            return
        issues = report.errors[:10]
        codes = tuple(sorted({issue.code for issue in report.errors}))
        contexts = "; ".join(
            f"{issue.code}(talep={issue.demand_id}, araç={issue.vehicle_id}, "
            f"tarih={issue.event_date}, değer={dict(issue.related_values)})"
            for issue in issues
        )
        raise BaselineBuildError(
            f"Başlangıç planı validator tarafından reddedildi: {contexts}",
            error_codes=codes,
            issues=issues,
        )


def _demand_sort_key(
    demand: DemandInstance,
) -> tuple[datetime, str]:
    """Order demands by ready timestamp then stable demand ID."""

    return (
        datetime.combine(demand.demand_date, demand.time_slot),
        demand.demand_id,
    )


__all__ = [
    "BaselineBuildError",
    "BaselineBuildResult",
    "ConservativeDirectBaselineBuilder",
]
