"""Build a same-date/same-origin two-destination greedy spot baseline.

``GreedySameOriginTwoStopBuilder`` preserves the mandatory-first behaviour and
atomic demand parts of the 6B strategy.  It evaluates two direct spot bins in
both possible delivery orders, commits only a strictly cheaper CostCalculator
result, and sends every unselected bin through the unchanged direct spot
scheduler.  This module is a deterministic baseline strategy, not a general
route optimizer, transfer model, vehicle-reuse chain or exporter.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from itertools import combinations
import math
from time import perf_counter

from src.domain import (
    DemandAssignment,
    DemandPart,
    HandlingOperation,
    TransportPlan,
    VehicleLeg,
    VehicleTrip,
    VehicleType,
)
from src.evaluation import CostCalculator, TransportPlanValidator

from .baseline import BaselineBuildError, BaselineBuildResult
from .handling_ledger import HandlingCapacityError, HandlingCapacityLedger
from .mandatory_filled import GreedyMandatoryRentalFilledBuilder
from .partitioning import partition_demands


@dataclass(frozen=True, slots=True)
class _SpotBin:
    """Hold one unchanged 6B same-O-D FFD box before physical scheduling."""

    index: int
    service_date: date
    origin: str
    destination: str
    ready_at: datetime
    parts: tuple[DemandPart, ...]
    total_desi: float


@dataclass(frozen=True, slots=True)
class _PairCandidate:
    """Identify two logical bins eligible for a two-destination trial."""

    first_bin: _SpotBin
    second_bin: _SpotBin

    @property
    def key(self) -> tuple[int, int]:
        """Return the stable source-bin key used for greedy exclusivity."""

        return (self.first_bin.index, self.second_bin.index)


@dataclass(frozen=True, slots=True)
class _PairEvaluation:
    """Keep one real scheduled/costed alternative and its transactional state."""

    candidate: _PairCandidate
    trip: VehicleTrip
    trial_ledger: HandlingCapacityLedger
    saving_tl: float
    combined_total_tl: float
    distance_km: float
    finish_at: datetime


@dataclass(frozen=True, slots=True)
class _BuildAttempt:
    """Bundle a validated plan with accepted pairs for economic rollback."""

    result: BaselineBuildResult
    selected: tuple[_PairEvaluation, ...]


class GreedySameOriginTwoStopBuilder(GreedyMandatoryRentalFilledBuilder):
    """Combine economical same-date/same-origin spot bins into two-stop trips.

    Args:
        problem: Static network data and standard immutable demand instances.
        policy: Shared desi/TL tolerances, handling duration, vehicle costing,
            SLA, mandatory-calendar and independent validation policy.

    Returns and connected layers:
        Inherited :meth:`build` returns a ``BaselineBuildResult``.  Mandatory
        filling and partitioning come from the 6B planning layer; transactional
        handling uses ``HandlingCapacityLedger``; every alternative and final
        plan is costed by ``CostCalculator`` and the final plan must pass
        ``TransportPlanValidator`` with zero errors.

    Strategy rules and failures:
        Only unchanged spot bins with the same demand date and origin but two
        different destinations are paired.  One non-Tır vehicle performs one
        load, two legs and two destination-specific unloads.  A pair is kept
        only when its real vehicle plus per-part SLA cost, including possible
        extra mandatory days, beats two direct trips beyond the configured TL
        tolerance.  Missing matrix arcs, same-day load failure, capacity
        failure, non-improving final cost or validator errors raise
        ``BaselineBuildError``; existing builders are not mutated.
    """

    def _build_result(self, started_at: float) -> BaselineBuildResult:
        """Build 6B reference, then return a strictly cheaper valid 6D plan."""

        reference = GreedyMandatoryRentalFilledBuilder(
            self.problem,
            self.policy,
        ).build()
        forbidden: set[tuple[int, int]] = set()
        while True:
            attempt = self._build_attempt(reference, forbidden)
            is_cheaper = (
                attempt.result.total_cost_tl
                < reference.total_cost_tl - self.policy.cost_tolerance_tl
            )
            if is_cheaper and attempt.selected:
                return self._with_elapsed_time(attempt.result, started_at)
            if not attempt.selected:
                raise BaselineBuildError(
                    "Aynı-origin iki-durak stratejisi kesin toplam tasarruf üretmedi.",
                    error_codes=("two_stop_no_strict_saving",),
                )
            weakest = min(
                attempt.selected,
                key=lambda item: (item.saving_tl, item.candidate.key),
            )
            forbidden.add(weakest.candidate.key)

    def _build_attempt(
        self,
        reference: BaselineBuildResult,
        forbidden: set[tuple[int, int]],
    ) -> _BuildAttempt:
        """Build one complete transactional greedy attempt from fresh inputs."""

        parts = self._partition_parts()
        ledger = self._empty_ledger()
        # Each attempt rebuilds from scratch, so the Tır ledger must reset too;
        # otherwise a retried attempt would see quota consumed by the previous
        # one and silently pack smaller bins.
        self._truck_ledger = self._new_truck_ledger()
        mandatory, assigned_ids = self._build_forecast_mandatory(parts, ledger)
        remaining = tuple(part for part in parts if part.part_id not in assigned_ids)
        bins = self._logical_spot_bins(remaining)
        calculator = CostCalculator(self.problem, self.policy)
        selected = self._select_pairs(
            bins,
            ledger,
            calculator,
            reference,
            forbidden,
        )
        used = {
            index
            for evaluation in selected
            for index in evaluation.candidate.key
        }
        direct = tuple(
            self._schedule_direct_bin(item, ledger)
            for item in bins
            if item.index not in used
        )
        spot = tuple(item.trip for item in selected) + direct
        result = self._finalize_plan(parts, mandatory, spot, ledger)
        return _BuildAttempt(result, selected)

    def _partition_parts(self) -> tuple[DemandPart, ...]:
        """Create exactly the same ordered atomic parts as the 6B strategy."""

        ordered = tuple(
            sorted(
                self.problem.demands,
                key=lambda demand: (
                    datetime.combine(demand.demand_date, demand.time_slot),
                    demand.demand_id,
                ),
            )
        )
        return partition_demands(
            ordered,
            self._maximum_non_truck_capacity(),
            self.policy.desi_tolerance,
        )

    def _empty_ledger(self) -> HandlingCapacityLedger:
        """Create the shared empty handling ledger used by one build attempt."""

        return HandlingCapacityLedger(
            self.problem.static.handling_capacities,
            self.policy.handling_minutes_per_desi,
            self.policy.desi_tolerance,
        )

    def _logical_spot_bins(
        self,
        parts: tuple[DemandPart, ...],
    ) -> tuple[_SpotBin, ...]:
        """Create unscheduled bins with the same grouping, FFD and IDs as 6A.

        Bin capacity follows the same Tır reservation used by 6A: a group whose
        daily Tır quota could be booked packs up to Tır capacity, every other
        group stays within the largest non-Tır capacity so a Kamyon can always
        serve it.
        """

        groups = self._group_parts(parts)
        truck_allowance = self._reserve_truck_allowance(groups)
        result: list[_SpotBin] = []
        index = 1
        for group_key in sorted(groups):
            group_parts = tuple(groups[group_key])
            for bin_parts in self._pack_group(
                group_parts,
                truck_allowance.get(group_key, 0),
            ):
                service_date, origin, destination = group_key
                result.append(
                    _SpotBin(
                        index,
                        service_date,
                        origin,
                        destination,
                        self._parts_ready_at(bin_parts),
                        bin_parts,
                        math.fsum(part.desi for part in bin_parts),
                    )
                )
                index += 1
        return tuple(result)

    def _select_pairs(
        self,
        bins: tuple[_SpotBin, ...],
        ledger: HandlingCapacityLedger,
        calculator: CostCalculator,
        reference: BaselineBuildResult,
        forbidden: set[tuple[int, int]],
    ) -> tuple[_PairEvaluation, ...]:
        """Rank preliminary savings, then commit disjoint real ledger trials."""

        candidates = tuple(
            item for item in self._candidate_pairs(bins)
            if item.key not in forbidden
        )
        scored: list[_PairEvaluation] = []
        for candidate in candidates:
            evaluation = self._evaluate_pair(
                candidate,
                ledger,
                calculator,
                reference,
            )
            if (
                evaluation is not None
                and evaluation.saving_tl > self.policy.cost_tolerance_tl
            ):
                scored.append(evaluation)
        used: set[int] = set()
        selected: list[_PairEvaluation] = []
        for preliminary in sorted(scored, key=self._evaluation_sort_key):
            if any(index in used for index in preliminary.candidate.key):
                continue
            actual = self._evaluate_pair(
                preliminary.candidate,
                ledger,
                calculator,
                reference,
            )
            if actual is None or actual.saving_tl <= self.policy.cost_tolerance_tl:
                continue
            ledger.commit_from(actual.trial_ledger)
            selected.append(actual)
            used.update(actual.candidate.key)
        return tuple(selected)

    def _candidate_pairs(
        self,
        bins: tuple[_SpotBin, ...],
    ) -> tuple[_PairCandidate, ...]:
        """Enumerate same-date/origin, different-destination feasible bin pairs."""

        grouped: dict[tuple[date, str], list[_SpotBin]] = defaultdict(list)
        for item in bins:
            grouped[(item.service_date, item.origin)].append(item)
        candidates: list[_PairCandidate] = []
        for key in sorted(grouped):
            ordered = sorted(grouped[key], key=lambda item: item.index)
            for first, second in combinations(ordered, 2):
                if first.destination == second.destination:
                    continue
                if not self._eligible_combined_vehicles(first, second):
                    continue
                if not self._both_stop_orders_have_arcs(first, second):
                    continue
                candidates.append(_PairCandidate(first, second))
        return tuple(candidates)

    def _eligible_combined_vehicles(
        self,
        first: _SpotBin,
        second: _SpotBin,
    ) -> tuple[VehicleType, ...]:
        """Return capacity-feasible non-Tır types without changing desi."""

        combined = first.total_desi + second.total_desi
        return tuple(
            sorted(
                (
                    vehicle
                    for vehicle in self._vehicle_types.values()
                    if vehicle.name != self.policy.truck_vehicle_type
                    and combined
                    <= vehicle.capacity_desi + self.policy.desi_tolerance
                ),
                key=lambda vehicle: (vehicle.capacity_desi, vehicle.name),
            )
        )

    def _both_stop_orders_have_arcs(
        self,
        first: _SpotBin,
        second: _SpotBin,
    ) -> bool:
        """Require all directed arcs needed to evaluate both stop orders."""

        required = (
            (first.origin, first.destination),
            (first.destination, second.destination),
            (first.origin, second.destination),
            (second.destination, first.destination),
        )
        return all(key in self._arcs for key in required)

    def _evaluate_pair(
        self,
        candidate: _PairCandidate,
        ledger: HandlingCapacityLedger,
        calculator: CostCalculator,
        reference: BaselineBuildResult,
    ) -> _PairEvaluation | None:
        """Compare two direct trips with every type/order on ledger forks."""

        direct_ledger = ledger.fork()
        try:
            direct_trips = tuple(
                self._schedule_direct_bin(item, direct_ledger)
                for item in sorted(
                    (candidate.first_bin, candidate.second_bin),
                    key=lambda value: value.index,
                )
            )
        except HandlingCapacityError:
            return None
        parts = self._pair_parts(candidate)
        direct_cost = calculator.calculate(TransportPlan(parts, direct_trips))
        direct_end = self._spot_completion_at(direct_trips)
        direct_total = direct_cost.total_cost_tl + self._horizon_surcharge(
            direct_end,
            reference,
        )
        options = self._combined_options(candidate, ledger, calculator, reference)
        if not options:
            return None
        best = min(options, key=self._combined_option_key)
        return _PairEvaluation(
            candidate,
            best.trip,
            best.trial_ledger,
            direct_total - best.combined_total_tl,
            best.combined_total_tl,
            best.distance_km,
            best.finish_at,
        )

    def _combined_options(
        self,
        candidate: _PairCandidate,
        ledger: HandlingCapacityLedger,
        calculator: CostCalculator,
        reference: BaselineBuildResult,
    ) -> tuple[_PairEvaluation, ...]:
        """Schedule and cost both stop orders for every feasible vehicle type."""

        options: list[_PairEvaluation] = []
        orders = (
            (candidate.first_bin, candidate.second_bin),
            (candidate.second_bin, candidate.first_bin),
        )
        for first_stop, second_stop in orders:
            for vehicle in self._eligible_combined_vehicles(*orders[0]):
                trial = ledger.fork()
                trip = self._schedule_two_stop(
                    candidate,
                    first_stop,
                    second_stop,
                    vehicle,
                    trial,
                )
                if trip is None:
                    continue
                cost = calculator.calculate(
                    TransportPlan(self._pair_parts(candidate), (trip,))
                )
                finish = self._spot_completion_at((trip,))
                total = cost.total_cost_tl + self._horizon_surcharge(
                    finish,
                    reference,
                )
                options.append(
                    _PairEvaluation(
                        candidate,
                        trip,
                        trial,
                        0.0,
                        total,
                        cost.vehicle_cost_lines[0].distance_km,
                        finish or trip.legs[-1].arrival_at,
                    )
                )
        return tuple(options)

    def _schedule_two_stop(
        self,
        candidate: _PairCandidate,
        first_stop: _SpotBin,
        second_stop: _SpotBin,
        vehicle: VehicleType,
        ledger: HandlingCapacityLedger,
    ) -> VehicleTrip | None:
        """Reserve one same-day load and two destination unloads transactionally."""

        total_desi = first_stop.total_desi + second_stop.total_desi
        ready_at = max(first_stop.ready_at, second_stop.ready_at)
        # No competition rule makes a spot vehicle load on its service day, and
        # the direct alternative this pair competes against already rolls to the
        # next day when origin capacity is full.  Requiring same-day loading
        # here rejected consolidation exactly where it is worth most: busy
        # origins on full days.  A later load is priced by CostCalculator and
        # _horizon_surcharge, so the strict-saving gate still decides.
        try:
            load = ledger.schedule(
                candidate.first_bin.origin,
                ready_at,
                total_desi,
            )
        except HandlingCapacityError:
            return None
        first_arc = self._arcs[(candidate.first_bin.origin, first_stop.destination)]
        second_arc = self._arcs[(first_stop.destination, second_stop.destination)]
        try:
            first_hours = first_arc.travel_hours_for(vehicle.name)
            second_hours = second_arc.travel_hours_for(vehicle.name)
        except KeyError:
            return None
        first_arrival = load[1] + timedelta(hours=first_hours)
        first_unload = ledger.schedule(
            first_stop.destination,
            first_arrival,
            first_stop.total_desi,
        )
        second_arrival = first_unload[1] + timedelta(hours=second_hours)
        second_unload = ledger.schedule(
            second_stop.destination,
            second_arrival,
            second_stop.total_desi,
        )
        return self._two_stop_models(
            candidate,
            first_stop,
            second_stop,
            vehicle,
            load,
            first_arrival,
            first_unload,
            second_arrival,
            second_unload,
        )

    @staticmethod
    def _two_stop_models(
        candidate: _PairCandidate,
        first_stop: _SpotBin,
        second_stop: _SpotBin,
        vehicle: VehicleType,
        load_interval: tuple[datetime, datetime],
        first_arrival: datetime,
        first_unload: tuple[datetime, datetime],
        second_arrival: datetime,
        second_unload: tuple[datetime, datetime],
    ) -> VehicleTrip:
        """Create one load, two legs and destination-specific final unloads."""

        stem = f"TSOD-{candidate.first_bin.index:05d}-{candidate.second_bin.index:05d}"
        vehicle_id = stem
        all_parts = first_stop.parts + second_stop.parts
        first_ids = tuple(part.part_id for part in first_stop.parts)
        second_ids = tuple(part.part_id for part in second_stop.parts)
        load = HandlingOperation(
            f"{stem}-LOAD",
            vehicle_id,
            first_stop.origin,
            "load",
            load_interval[0],
            load_interval[1],
            math.fsum(part.desi for part in all_parts),
            tuple(part.part_id for part in all_parts),
        )
        unload_one = HandlingOperation(
            f"{stem}-UNLOAD-1", vehicle_id, first_stop.destination, "unload",
            first_unload[0], first_unload[1], first_stop.total_desi, first_ids,
        )
        unload_two = HandlingOperation(
            f"{stem}-UNLOAD-2", vehicle_id, second_stop.destination, "unload",
            second_unload[0], second_unload[1], second_stop.total_desi, second_ids,
        )
        first_leg, second_leg = GreedySameOriginTwoStopBuilder._two_stop_legs(
            stem, first_stop, second_stop, load, unload_one, unload_two,
            first_arrival, second_arrival,
        )
        return VehicleTrip(
            f"{stem}-TRIP", vehicle_id, vehicle.name, "spot",
            (first_leg, second_leg), (load, unload_one, unload_two),
        )

    @staticmethod
    def _two_stop_legs(
        stem: str,
        first_stop: _SpotBin,
        second_stop: _SpotBin,
        load: HandlingOperation,
        unload_one: HandlingOperation,
        unload_two: HandlingOperation,
        first_arrival: datetime,
        second_arrival: datetime,
    ) -> tuple[VehicleLeg, VehicleLeg]:
        """Build carried-load assignments that shrink after the first stop."""

        first_assignments = tuple(
            DemandAssignment(part.part_id, part.desi, True, True, True)
            for part in first_stop.parts
        ) + tuple(
            DemandAssignment(part.part_id, part.desi, True, False, False)
            for part in second_stop.parts
        )
        second_assignments = tuple(
            DemandAssignment(part.part_id, part.desi, False, True, True)
            for part in second_stop.parts
        )
        first_leg = VehicleLeg(
            f"{stem}-LEG-1", first_stop.origin, first_stop.destination,
            load.end_at, first_arrival, first_assignments,
            load.operation_id, unload_one.operation_id,
        )
        second_leg = VehicleLeg(
            f"{stem}-LEG-2", first_stop.destination, second_stop.destination,
            unload_one.end_at, second_arrival, second_assignments,
            None, unload_two.operation_id,
        )
        return first_leg, second_leg

    def _schedule_direct_bin(
        self,
        item: _SpotBin,
        ledger: HandlingCapacityLedger,
    ) -> VehicleTrip:
        """Send one unselected logical bin through unchanged 6A scheduling."""

        return self._build_bin_trip(
            item.index,
            (item.service_date, item.origin, item.destination),
            item.ready_at,
            item.parts,
            ledger,
        )

    @staticmethod
    def _pair_parts(candidate: _PairCandidate) -> tuple[DemandPart, ...]:
        """Return both bins' unchanged atoms in stable source-bin order."""

        return candidate.first_bin.parts + candidate.second_bin.parts

    def _horizon_surcharge(
        self,
        completion_at: datetime | None,
        reference: BaselineBuildResult,
    ) -> float:
        """Price empty mandatory days beyond the validated 6B horizon in TL."""

        if completion_at is None:
            return 0.0
        reference_date = max(
            self.policy.mandatory_forecast_end_date,
            reference.completion_at.date()
            if reference.completion_at is not None
            else self.policy.mandatory_forecast_end_date,
        )
        completion_date = max(
            self.policy.mandatory_forecast_end_date,
            completion_at.date(),
        )
        extra_days = max(0, (completion_date - reference_date).days)
        return extra_days * self._empty_mandatory_daily_cost()

    def _empty_mandatory_daily_cost(self) -> float:
        """Return one full empty mandatory schedule day's exact vehicle TL."""

        total = 0.0
        for route in self.problem.static.mandatory_rental_routes:
            arc = self._arcs[(route.origin, route.destination)]
            vehicle = self._vehicle_types[route.vehicle_type]
            usage_hours = arc.travel_hours_for(route.vehicle_type)
            one_vehicle = (
                math.ceil(usage_hours) * vehicle.rented_hourly_cost_tl
                + arc.distance_km * vehicle.rented_cost_per_km_tl
            )
            total += route.vehicle_count * one_vehicle
        return total

    @staticmethod
    def _combined_option_key(
        item: _PairEvaluation,
    ) -> tuple[float, float, datetime, str, str, str]:
        """Choose lowest TL, km, completion, route and type deterministically."""

        destinations = tuple(leg.destination for leg in item.trip.legs)
        return (
            item.combined_total_tl,
            item.distance_km,
            item.finish_at,
            destinations[0],
            destinations[1],
            item.trip.vehicle_type,
        )

    @staticmethod
    def _evaluation_sort_key(item: _PairEvaluation) -> tuple:
        """Apply the documented greedy saving/cost/km/time and ID priorities."""

        candidate = item.candidate
        destinations = tuple(leg.destination for leg in item.trip.legs)
        return (
            -item.saving_tl,
            item.combined_total_tl,
            item.distance_km,
            item.finish_at,
            candidate.first_bin.service_date,
            candidate.first_bin.origin,
            destinations[0],
            destinations[1],
            candidate.first_bin.index,
            candidate.second_bin.index,
        )

    def _finalize_plan(
        self,
        parts: tuple[DemandPart, ...],
        forecast_mandatory: tuple[VehicleTrip, ...],
        spot: tuple[VehicleTrip, ...],
        ledger: HandlingCapacityLedger,
    ) -> BaselineBuildResult:
        """Extend mandatory horizon, cost all activities and require validation."""

        completion_at = self._spot_completion_at(forecast_mandatory + spot)
        continuation = self._build_empty_continuation(completion_at)
        mandatory = forecast_mandatory + continuation
        draft = TransportPlan(parts, mandatory + spot)
        cost = CostCalculator(self.problem, self.policy).calculate(draft)
        plan = TransportPlan(parts, draft.trips, reported_cost=cost)
        report = TransportPlanValidator(self.problem, self.policy).validate(plan)
        self._require_valid_report(report)
        return BaselineBuildResult(
            plan,
            report,
            cost.total_cost_tl,
            completion_at,
            len(parts),
            len(spot),
            len(mandatory),
            ledger.snapshot(),
            0.0,
        )

    @staticmethod
    def _with_elapsed_time(
        result: BaselineBuildResult,
        started_at: float,
    ) -> BaselineBuildResult:
        """Return the accepted immutable result with total wall-clock seconds."""

        return BaselineBuildResult(
            result.plan,
            result.validation_report,
            result.total_cost_tl,
            result.completion_at,
            result.demand_part_count,
            result.spot_vehicle_count,
            result.mandatory_vehicle_count,
            result.handling_usage,
            perf_counter() - started_at,
        )


__all__ = ["GreedySameOriginTwoStopBuilder"]
